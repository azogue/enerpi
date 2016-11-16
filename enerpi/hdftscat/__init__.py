# -*- coding: utf-8 -*-
# import asyncio
# import concurrent.futures as futures
import glob
import json
import numpy as np
import os
import pandas as pd
import re
import shutil
from enerpi.base import timeit, log
# from memory_profiler import profile


ARCHIVE_AUTO = 0
ARCHIVE_HOURLY = 1
ARCHIVE_DAILY = 2
ARCHIVE_MONTHLY = 3

STORE_EXT = '.h5'
KWARGS_SAVE = dict(complevel=9, complib='blosc', fletcher32=True)

DIR_CURRENT_MONTH = 'CURRENT_MONTH'
DIR_BACKUP = 'OLD_STORES'
ST_TODAY = os.path.join(DIR_CURRENT_MONTH, 'TODAY' + STORE_EXT)
INDEX = 'data_catalog.csv'

YEAR_MASK = '{}_YEAR_{:%Y}'
RG_YEAR_MASK = re.compile('(?P<name>\.*)_YEAR_(?P<year>\d{4})')
MONTH_MASK = '{}_{:%Y_MONTH_%m}' + STORE_EXT
RG_MONTH_MASK = re.compile('(?P<name>\.*)_(?P<year>\d{4})_MONTH_(?P<month>\d{2})')
DAY_MASK = '{}_{:%Y_%m_DAY_%d}' + STORE_EXT
RG_DAY_MASK = re.compile('(?P<name>\.*)_(?P<year>\d{4})_(?P<month>\d{2})_DAY_(?P<day>\d{2})')


def _get_time_slice(start=None, end=None, last_hours=None, min_ts=None):
    if last_hours is not None:
        if type(last_hours) is str:
            last_hours = int(last_hours)
        start = pd.Timestamp.now().replace(minute=0, second=0, microsecond=0) - pd.Timedelta(hours=last_hours)
        return start, None
    else:
        if (start is None) and (min_ts is not None):
            start = min_ts
        if start is np.nan:
            start = pd.Timestamp.now() - pd.Timedelta(hours=1)
        return start, end


def _concat_loaded_data(dataframes, ini, fin=None, verbose=False):
    try:
        valid_dfs = list(filter(lambda df: df is not None, dataframes))
        if valid_dfs:
            dataframe = pd.DataFrame(pd.concat(valid_dfs)).sort_index()
            if fin is not None:
                return dataframe.loc[ini:fin]
            return dataframe.loc[ini:]
        else:
            log('GET DATA -> No valid dfs ({}): {})'.format(len(dataframes), dataframes), 'warn', verbose)
    except ValueError as e:
        log('GET DATA ERROR: {}'.format(e), 'error', verbose)
    return None


# TODO to_json / from_json para recoger info vía webapi y replicarla en un catálogo remoto


class HDFTimeSeriesCatalog(object):
    """
    Clase para gestionar la base de datos generada por ENERPI Logger.
    Esta DB se gestiona mediante pandas HDFStores comprimidos y separados por meses, con la siguiente estructura:

    DATA_PATH
        YEAR_XXXX
            MONTH_XX.h5     data_processed, data_summary, extra_summary
            MONTH_XY.h5     data_processed, data_summary, extra_summary
            ...
        YEAR_XXXX
            MONTH_XY.h5     data_processed, data_summary, extra_summary
            ...
        CURRENT_MONTH
            DAY_XX.h5       data_processed, data_summary
            DAY_XX.h5       data_processed, data_summary
            ...
        TODAY
            TODAY.h5        data_processed
        TEMP_DATA.h5        data_raw

        OLD_STORES
            BKP_OLD_1
            ...

    * Reconstrucción de índice:
        Lee todos los store, los copia a BACKUP_ORIG si procede, crea estructura de árbol de archivo,
         distribuye data y guarda stores en árbol. Borra originales.
        * Recontrucción parcial: ajusta sólo los nuevos stores creados (para ejecutar al inicio). Sin backup
    * Lectura de datos:
        De t_0 a t_f(=now). Genera paths de stores del intervalo requerido, lee los stores (en async),
        concatena la información, y devuelve.
        Si se lee con t_f > TODAY, se genera data_processed y/o data_summary para el último tramo (RAW data)
    * Archivado periódico:
        0. Cada (+-) hora, Acumulación de TEMP_DATA en TODAY:
            lee RAW TEMP_DATA, procesa, append nuevos datos a TODAY --> Devuelve señal de borrado de TEMP_DATA.
                **process_data**
        1. Al cierre del día:
            lee TODAY, GENERA SUMMARY, escribe DAY_XX, Limpia TODAY.
                **process_data_summary**
        2. Al cierre de mes:
            lee todos los DAY_XX de CURRENT_MONTH, concatena, GENERA EXTRA SUMMARY, escribe MONTH_XX,
            Borra stores leídos.
                **process_data_summary_extra**
        3. (OPC). Cierre Anual.

    La razón de ser de este módulo, en vez de recurrir a una DB tradicional (sqlite, mysql, ...), es optimizar tanto
    el tamaño ocupado por los datos (considerablemente menor con HDF comprimido), como las velocidades de acceso,
    el nº de escrituras; así como minimizar la pérdida de datos en caso de corrupción del soporte físico.

    Ténganse en cuenta las muy concretas necesidades de operación de este módulo, pensado para ejecutarse en una
    Raspberry PI con raspbian sobre una tarjeta SD de 8/16 GB, con un mini-webserver para el acceso en red local desde
    no más de 2/3 clientes. En esa situación, una vez que el archivo de datos es considerable, el uso de ficheros
    individuales no ralentiza en absoluto los tiempos de 'query', además de facilitar considerablemente el
    backup incremental de los datos adquiridos.
    """

    def __init__(self,
                 base_path=os.getcwd(),
                 preffix='DATA',
                 raw_file='temp' + STORE_EXT,
                 key_raw_data='/raw_data',
                 key_summary_data='/hours',
                 key_summary_extra='/days',
                 catalog_file=INDEX,
                 check_integrity=True,
                 archive_existent=False,
                 verbose=True):
        self.base_path = os.path.abspath(base_path)
        self.name = preffix
        self.verbose = verbose

        self.raw_store = raw_file + STORE_EXT if not raw_file.endswith(STORE_EXT) else raw_file
        self.key_raw = key_raw_data
        self.key_summary = key_summary_data
        self.key_summary_extra = key_summary_extra

        # Index:
        self.catalog_file = catalog_file
        self.min_ts = None
        if archive_existent:
            self.update_catalog()
        else:
            self.tree = self._get_index(check_index=check_integrity)
        self.min_ts = self.tree['ts_ini'].min() if self._exist() else np.nan
        self.index_ts = self._ts_filepath(self.catalog_file)

    @staticmethod
    def is_raw_data(data):
        """
        Is RAW Data? If it is, call process_data before storing it. Implement on subclass
        :param data:
        :return: :bool:
        """
        raise NotImplementedError

    @staticmethod
    def process_data(data):
        """
        RAW Data to processed data. Implement on subclass
        :param data:
        :return: data_p
        """
        raise NotImplementedError

    @staticmethod
    def process_data_summary(data):
        """
        From Processed data to Processed data + summary. Implement on subclass
        :param data:
        :return: data, data_s
        """
        raise NotImplementedError

    @staticmethod
    def process_data_summary_extra(data):
        """
        From Processed data to Processed data + summary + summary extra. Implement on subclass
        :param data:
        :return: data, data_s, data_se
        """
        raise NotImplementedError

    def _exist(self):
        return self.tree is not None and not self.tree.empty

    def __repr__(self):
        mints = '{:%B/%Y}'.format(self.min_ts) if self._exist() else '--'
        idxts = '{:%d/%m/%y %H:%M:%S}'.format(self.index_ts) if self._exist() else '--'
        cad = '<HDFCatalog[{}] ->{}; From:{}; last_upd={}>\n\tINDEX [{}]:\n{}\n'
        catalog = self.tree.sort_values(by=['is_cat', 'key', 'ts_fin']) if self._exist() else '\t\t*NO INDEX*'
        return cad.format(self.name, self.base_path, mints, idxts, self.catalog_file, catalog)

    def _ts_filepath(self, rel_path):
        if self.base_path not in rel_path:
            p = os.path.join(self.base_path, rel_path)
        else:
            p = rel_path
        return pd.Timestamp.fromtimestamp(os.path.getmtime(p)) if os.path.exists(p) else None

    def _load_index(self):
        p = os.path.join(self.base_path, self.catalog_file)
        try:
            index = pd.read_csv(p, index_col=0, parse_dates=['ts_ini', 'ts_fin', 'ts_st'])
            index.cols = index.cols.map(lambda x: json.loads(x.replace("'", '"')))
            return index.drop_duplicates(subset=['st', 'key', 'ts_st', 'ts_ini', 'ts_fin'])
        except FileNotFoundError:
            log('FileNotFoundError reading HDFTSCAT INDEX in "{}". Rebuilding index...'
                .format(p), 'error', self.verbose)
        except AttributeError:
            log('AttributeError reading HDFTSCAT INDEX in "{}". Corrupted?\n{}\n ...Rebuilding index...'
                .format(p, open(p, 'r').readlines()), 'error', self.verbose)
        return None

    def _get_index(self, check_index=True):
        """
        Loading and checking of the catalog index (It's contained in a CSV file in the catalog root directory).
        ** The index is updated or rebuilt if necessary.
        :return: index, Pandas DataFrame with the catalog of all included HDF Stores.
        """

        index = self._load_index()
        if index is None:
            index = self._make_index(distribute_existent=True)
            if index is not None and not index.empty:
                self._save_index(index)
        elif check_index:
            index = self._check_index(index)
        return index

    def _save_index(self, index):
        p = os.path.join(self.base_path, self.catalog_file)
        index.to_csv(p)
        return True

    def _check_index(self, index):
        index = index.copy()
        paths = index['st'].apply(lambda x: os.path.join(self.base_path, x))
        times = [self._ts_filepath(p) for p in paths]
        pb_bkp = os.path.join(self.base_path, DIR_BACKUP)
        new_stores = [f.replace(self.base_path + os.path.sep, '')
                      for f in glob.glob(os.path.join(self.base_path, '**'), recursive=True)
                      if (f.endswith(STORE_EXT) and (pb_bkp not in f) and
                          (f != self.raw_store) and (f not in paths.values))]
        index['new_ts'] = times
        b_no_existe = index['new_ts'].isnull()
        index.loc[b_no_existe, 'modif'] = True
        if not b_no_existe.all():
            index.loc[~b_no_existe, 'modif'] = index.loc[~b_no_existe, 'ts_st'] != index.loc[~b_no_existe, 'new_ts']
        if new_stores:
            log('NEW STORES WERE FOUND: {}'.format(new_stores), 'warn', self.verbose)
        new_stores += paths.loc[index['new_ts'].notnull() & index['modif']].drop_duplicates().tolist()
        cond_no_change = index['new_ts'].notnull() & ~index['modif']
        index = index[cond_no_change].drop(['modif', 'ts_st'], axis=1).rename(columns={'new_ts': 'ts_st'})
        lost_stores = index.loc[index['ts_st'].isnull(), 'st'].drop_duplicates().tolist()
        if new_stores:
            log('THERE ARE MODIFIED STORES. INDEX WILL BE RE-CREATED FOR THESE: {}'.format(new_stores),
                'debug', self.verbose)
            new_rows = self._make_index(distribute_existent=True, paths=new_stores)
            index = index.set_index('st').drop(new_rows['st'].drop_duplicates().values, errors='ignore').reset_index()
            index = pd.DataFrame(pd.concat([index, new_rows])).sort_values(by='ts_ini')
        if lost_stores:
            log('SOME STORES WERE LOST. THESE WILL BE REMOVED FROM INDEX: {}'.format(lost_stores), 'warn', self.verbose)
            index = index.set_index('st').drop(lost_stores, errors='ignore').sort_values(by='ts_ini').reset_index()
        self._save_index(index)
        # if new_stores or lost_stores:
        self.min_ts = index['ts_ini'].min() if not index.empty else np.nan
        self.index_ts = self._ts_filepath(self.catalog_file)
        return index

    def _load_hdf(self, rel_path, key=None, func_store=None, columns=None):
        p = os.path.join(self.base_path, rel_path)
        k = key or self.key_raw
        try:
            with pd.HDFStore(p, mode='r') as st:
                if func_store is None:
                    if columns is None:
                        data = st[k]
                    else:
                        data = st.select(k, columns=columns)
                else:
                    data = func_store(st)
            return data
        except KeyError as e:
            log('load_hdf KEYERROR -> ST:"{}", KEY:{}; -> {}'.format(p, k, e), 'error', self.verbose)
        except OSError as e:
            log('load_hdf OSERROR -> ST:"{}", KEY:"{}"'
                .format(p, k, e, e.__class__, os.path.exists(p)), 'debug', self.verbose)
        return None

    def _save_hdf(self, data, path, key, mode='a', **kwargs):
        p = os.path.join(self.base_path, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with pd.HDFStore(p, mode=mode, **kwargs) as st:
            try:
                if (type(key) is str) and (type(data) is pd.DataFrame):
                    st.append(key, data)
                else:
                    for k, df in zip(key, data):
                        st.append(k, df)
            except ValueError as e:
                log('ERROR en _save_hdf: {}'.format(e), 'error', self.verbose)
                log(st.keys(), 'debug', self.verbose)
                assert()
        log('STORE "{}", "{}"\t->\t{:.1f} KB'.format(path, key, os.path.getsize(p) / 1000), 'debug', self.verbose)
        return True

    def _make_index_path(self, ts, w_day=False):
        if ts.date() == pd.Timestamp.today().date():
            p = ST_TODAY
        elif w_day:
            p = os.path.join(DIR_CURRENT_MONTH, DAY_MASK.format(self.name, ts))
        else:
            p = os.path.join(YEAR_MASK.format(self.name, ts), MONTH_MASK.format(self.name, ts))
        return p

    def _get_paths(self, start=None, end=None, last_hours=None):

        def _get_paths_interval(ts_ini, ts_fin=None):
            ahora = pd.Timestamp.now()
            ts_ini = pd.Timestamp(ts_ini)
            ts_fin = pd.Timestamp(ts_fin) if ts_fin else ahora
            periods = (ts_fin.year * 12 + ts_fin.month) - (ts_ini.year * 12 + ts_ini.month)
            index = pd.DatetimeIndex(freq='M', start=ts_ini.date(), periods=periods + 1)
            paths = []
            for i in index:
                if (ahora.year == i.year) and (ahora.month == i.month):
                    init = ahora.replace(day=1).date() if len(paths) > 0 else ts_ini.date()
                    index_d = pd.DatetimeIndex(freq='D', start=init, periods=ts_fin.day - init.day + 1)
                    [paths.append(self._make_index_path(i, w_day=True)) for i in index_d]
                else:
                    paths.append(self._make_index_path(i, w_day=False))
            return paths

        t0, tf = _get_time_slice(start, end, last_hours, min_ts=self.min_ts)
        return _get_paths_interval(ts_ini=t0, ts_fin=tf)

    def _load_today(self):
        return self._load_hdf(ST_TODAY, key=self.key_raw)

    def _load_current_month(self, with_summary_data=True):
        days_cm = list(sorted([p.replace(self.base_path + os.path.sep, '')
                               for p in glob.glob(os.path.join(self.base_path, DIR_CURRENT_MONTH,
                                                               '*{}'.format(STORE_EXT)))]))
        if with_summary_data:
            extracted = [self._load_hdf(p, func_store=lambda st: (st[self.key_raw], st[self.key_summary]))
                         for p in days_cm]
            df = pd.DataFrame(pd.concat([e[0] for e in extracted if e is not None]))
            df_s = pd.DataFrame(pd.concat([e[1] for e in extracted if e is not None]))
            return df, df_s, days_cm
        else:
            if days_cm:
                df = pd.DataFrame(pd.concat([self._load_hdf(p, key=self.key_raw) for p in days_cm]))
                return df, days_cm
            return None, []

    def _classify_data(self, df, func_save_data):
        paths = []
        ahora = pd.Timestamp.now()
        gb_años = df.groupby(pd.TimeGrouper(freq='A'))
        for ts_year, d_year in gb_años:
            if not d_year.empty:
                gb_meses = d_year.groupby(pd.TimeGrouper(freq='M'))
                for ts_month, d_month in gb_meses:
                    if not d_month.empty:
                        if (ts_year.year == ahora.year) and (ts_month.month == ahora.month):
                            # CURRENT MONTH
                            gb_dias = d_month.groupby(pd.TimeGrouper(freq='D', closed='left', label='left'))
                            for ts_day, d_day in gb_dias:
                                if not d_day.empty:
                                    if ts_day.day == ahora.day:
                                        # TODAY
                                        func_save_data(ST_TODAY, d_day, None, None)
                                        paths.append(ST_TODAY)
                                    else:
                                        # ARCHIVE DAY
                                        p = self._make_index_path(ts_day, w_day=True)
                                        log('# ARCHIVE DAY {:%Y-%m-%d} -> {}'.format(ts_day, p), 'debug', self.verbose)
                                        d_day_p, c_day = self.process_data_summary(d_day)
                                        func_save_data(p, d_day_p, c_day, None)
                                        paths.append(p)
                        else:
                            # ARCHIVE MONTH
                            p = self._make_index_path(ts_month, w_day=False)
                            log('# ARCHIVE MONTH --> {}. GOING TO process_data_summary_extra'.format(p),
                                'debug', self.verbose)
                            d_month_p, c_month, c_month_extra = self.process_data_summary_extra(d_month)
                            func_save_data(p, d_month_p, c_month, c_month_extra)
                            paths.append(p)
        return list(sorted(paths))

    def _is_catalog_path(self, st, ts_ini, ts_fin):
        if st == ST_TODAY:
            return True
        elif st == self.raw_store:
            return False

        paths = self._get_paths(ts_ini, ts_fin)
        if (len(paths) == 1) and (paths[0] == st):
            return True
        return False

    def _gen_index_entries(self, paths=None):

        def _get_frame_data(store):
            return [(relat_path, key, store.select(key, stop=1).index[0], store.select(key, start=-1).index[0],
                     st_mtime, store[key].shape[0], key == self.key_raw, list(store[key].columns))
                    for key in store.keys()]

        dataframes = []
        pb_bkp = os.path.join(self.base_path, DIR_BACKUP)
        if paths is None:
            paths = glob.glob(os.path.join(self.base_path, '**'), recursive=True)
        for f in paths:
            if f.endswith(STORE_EXT) and (pb_bkp not in f) and (f != self.raw_store):
                relat_path = f.replace(self.base_path + os.path.sep, '')
                st_mtime = self._ts_filepath(relat_path)
                new = self._load_hdf(f, self.key_raw, func_store=_get_frame_data)
                if new:
                    dataframes += new
        df = pd.DataFrame(dataframes, columns=['st', 'key', 'ts_ini', 'ts_fin', 'ts_st', 'n_rows', 'is_raw', 'cols']
                          ).sort_values(by='ts_ini')
        if not df.empty:
            claves = df[df.is_raw].groupby('st').first()
            iscat = pd.DataFrame(claves.apply(lambda x: self._is_catalog_path(x.name, x['ts_ini'], x['ts_fin']),
                                              axis=1).rename('is_cat'))
            if not iscat.empty:
                return df.set_index('st').join(iscat).fillna(False).sort_values(by='ts_ini').reset_index()
        return df.T.append(pd.Series([], name='ts_cat').T).T

    # @timeit('_make_index')
    # @profile
    def _make_index(self, distribute_existent=True, paths=None):
        df = self._gen_index_entries(paths=paths)
        if distribute_existent and not df.empty:
            if not df[df.is_raw & ~df.is_cat].empty:
                raw_to_distr = df[df.is_raw & ~df.is_cat]
                # log('Distribuyendo datos desde:\n{}'.format(raw_to_distr), 'debug', self.verbose)
                data = pd.DataFrame(pd.concat([self._load_hdf(p, key=self.key_raw)
                                               for p in raw_to_distr['st']])).sort_index()
                if self.is_raw_data(data):
                    data = self.process_data(data)
                mod_paths = self._distribute_data(data, mode='a')
                for p in raw_to_distr['st']:
                    p_bkp = os.path.join(self.base_path, DIR_BACKUP, p)
                    os.makedirs(os.path.dirname(p_bkp), exist_ok=True)
                    shutil.copyfile(os.path.join(self.base_path, p), p_bkp)
                    os.remove(os.path.join(self.base_path, p))
                log('2º PASADA con modpaths', 'debug', self.verbose)
                df = df.set_index('st').drop(mod_paths, errors='ignore').reset_index()
                df_2 = self._gen_index_entries(paths=mod_paths)
                df = pd.concat([df.drop(raw_to_distr.index, errors='ignore'), df_2])
            else:
                log('No hay stores que distribuir', 'debug', self.verbose)
            return df
        elif not df.empty:
            return df
        return None

    def _load_store(self, path_idx, with_summary=False, column=None):

        def _get_data_from_store(st):
            d1 = st[self.key_raw]
            try:
                d2 = st[self.key_summary]
            except KeyError:
                d1, d2 = self.process_data_summary(d1)
            return d1, d2

        # asyncio.sleep(0)
        if with_summary:
            extracted = self._load_hdf(path_idx, func_store=_get_data_from_store)
        elif column is not None:
            extracted = self._load_hdf(path_idx, key=self.key_raw, columns=[column])
        else:
            extracted = self._load_hdf(path_idx, key=self.key_raw)
        # asyncio.sleep(0)
        if (extracted is None) and with_summary:
            return None, None
        return extracted

    def _load_summary(self, path_idx):

        def _get_summary_from_store(st):
            try:
                d2 = st[self.key_summary]
            except KeyError:
                _, d2 = self.process_data_summary(st[self.key_raw])
            return d2

        # asyncio.sleep(0)
        return self._load_hdf(path_idx, func_store=_get_summary_from_store)

    def _remove_old_if_archive(self, old_stores, new_stores, ahora):
        self.tree = self.tree.set_index('st').drop(ST_TODAY, errors='ignore').reset_index()
        for p in old_stores:
            remove = False
            if ST_TODAY == p:
                if ST_TODAY not in new_stores:
                    remove = True
            elif int(RG_DAY_MASK.search(p).groupdict()['month']) != ahora.month:
                remove = True
            if remove:
                p_bkp = os.path.join(self.base_path, DIR_BACKUP, p)
                os.makedirs(os.path.dirname(p_bkp), exist_ok=True)
                shutil.copyfile(os.path.join(self.base_path, p), p_bkp)
                os.remove(os.path.join(self.base_path, p))

    def archive_periodic(self, new_data=None, reload_index=False):
        """
        * Archivado periódico:
            0. Cada hora, Acumulación de TEMP_DATA en TODAY:
                lee TEMP_DATA, procesa, append nuevos datos a TODAY --> Devuelve señal de borrado de TEMP_DATA.
                    **process_data**
            1. Al cierre del día:
                lee TODAY, GENERA SUMMARY, escribe DAY_XX, Limpia TODAY.
                    **process_data_summary**
            2. Al cierre de mes:
                lee todos los DAY_XX de CURRENT_MONTH, concatena, GENERA EXTRA SUMMARY, escribe MONTH_XX, y borra DAY's.
                    **process_data_summary_extra**
            3. (OPC). Cierre Anual.

        :param new_data:
        :param reload_index:
        """

        if reload_index:
            self.tree = self._get_index(check_index=False)
        new_stores = []
        ahora = pd.Timestamp.now()
        month_now = ahora.year * 12 + ahora.month
        monthly_archive = False
        new_data_append = False
        if new_data is not None:
            new_data_append = True
            log('ARCHIVE NEW RAW DATA: {}'.format(new_data.shape), 'debug', self.verbose)
            if self.is_raw_data(new_data):
                new_data = self.process_data(new_data)
            hay_cambio_dia = new_data.index[-1].day != new_data.index[0].day
            hay_cambio_mes = new_data.index[-1].month != new_data.index[0].month
            if hay_cambio_mes:
                monthly_archive = True
                month, old_stores = self._load_current_month(with_summary_data=False)
                log('** ARCHIVE MONTH: {}, SHAPE: {}'
                    .format(old_stores, month.shape if month is not None else 'None'), 'info', self.verbose)
                if month is not None:
                    new_data = month.append(new_data)
                new_stores += self._distribute_data(new_data, mode='w')
                self._remove_old_if_archive(old_stores, new_stores, ahora)
            elif hay_cambio_dia:
                today = self._load_today()
                log('** ARCHIVE DAY', 'info', self.verbose)
                if today is not None:
                    new_data = pd.DataFrame(pd.concat([today, new_data])).sort_index().groupby(level=0).first()
                new_stores += self._distribute_data(new_data, mode='w')
                self._remove_old_if_archive([ST_TODAY], new_stores, ahora)
            else:
                new_stores += self._distribute_data(new_data, mode='a')
        else:
            data_current_month = self.tree[self.tree['st'].str.contains(DIR_CURRENT_MONTH)] if self._exist() else None
            if data_current_month is not None and not data_current_month.empty:
                # Archivo mensual
                month_ts = data_current_month['ts_ini'].min()
                month_cm = month_ts.year * 12 + month_ts.month
                if not data_current_month.empty and (month_cm < month_now):
                    month, days_cm = self._load_current_month(with_summary_data=False)
                    log('** ARCHIVE MONTH: {}'.format(days_cm), 'info', self.verbose)
                    new_stores += self._distribute_data(month, mode='a')
                    self._remove_old_if_archive(days_cm, new_stores, ahora)
                    monthly_archive = len(new_stores) > 0
            ts_today = self._ts_filepath(ST_TODAY)
            if not monthly_archive and ts_today:
                day_now = ahora.toordinal()
                try:
                    today_min = self.tree.set_index('st').loc[ST_TODAY, 'ts_ini'].toordinal()
                except AttributeError as e:
                    log('AttributeError getting today_ts_min ({}). Tree:\n{}'
                        .format(e, self.tree), 'error', self.verbose)
                    today_min = self.tree.set_index('st').loc[ST_TODAY, 'ts_ini'].ix[0].toordinal()
                if (today_min < day_now) or (ts_today.toordinal() < day_now):
                    log('ARCHIVE DAY', 'debug', self.verbose)
                    today = self._load_today()
                    new_stores += self._distribute_data(today, mode='a')
                    self._remove_old_if_archive([ST_TODAY], new_stores, ahora)

        if new_stores:
            new_stores = list(set(new_stores))
            log('Modificando index en archive_periodic: new_stores: {}'.format(new_stores), 'debug', self.verbose)
            new_rows = self._make_index(distribute_existent=False, paths=new_stores)
            if self.tree is not None:
                index = self.tree.copy()
                index = index.set_index('st').drop(new_rows['st'].drop_duplicates().values, errors='ignore'
                                                   ).reset_index()
                index = pd.DataFrame(pd.concat([index, new_rows])).sort_values(by='ts_ini')
            else:
                index = pd.DataFrame(new_rows).sort_values(by='ts_ini')
            if monthly_archive:
                index = self._check_index(index)
            self.tree = index
            self._save_index(index)
            self.min_ts = self.tree['ts_ini'].min() if not self.tree.empty else np.nan
            self.index_ts = self._ts_filepath(self.catalog_file)
            return True
        return new_data_append

    @timeit('update_catalog')
    def update_catalog(self, data=None):
        """
        Execute this function periodically, within the raw data generator process, to maintain the catalog updated.
        This function reads the temporal data store, adds it to the catalog, and deletes (recreates) the temporal store.
        """
        temp_data = data if data is not None else self._load_hdf(self.raw_store, self.key_raw)
        new_data = self.archive_periodic(new_data=temp_data, reload_index=True)
        if new_data:
            p = os.path.join(self.base_path, self.raw_store)
            with pd.HDFStore(p, 'w'):
                info = 'Temporal data has been archived. Reset of "{}" is done. Store new size: {:.1f} KB'
                log(info.format(p, os.path.getsize(p) / 1000), 'debug', self.verbose)
            return True
        return False

    def _distribute_data(self, data, mode='a'):

        def _save_distributed_data(p, d1, d2, d3):
            f = list(filter(lambda x: x[0] is not None, zip([d1, d2, d3],
                                                            [self.key_raw, self.key_summary, self.key_summary_extra])))
            dfs, keys = list(zip(*f))[0], list(zip(*f))[1]
            p_abs = os.path.join(self.base_path, p)
            if mode == 'a' and os.path.exists(p_abs):  # 1º se lee, se concatena, y se eliminan duplicados
                log('** Leyendo información previa del STORE: {}'.format(p_abs), 'debug', self.verbose)
                old_dfs = self._load_hdf(p, func_store=lambda st: [st[k] for k in keys])
                dfs = [pd.DataFrame(pd.concat([old, df])
                                    ).sort_index().reset_index().drop_duplicates(subset='ts').set_index('ts')
                       for old, df in zip(old_dfs, dfs)]
            self._save_hdf(dfs, p, keys, mode='w', **KWARGS_SAVE)

        paths = self._classify_data(data, _save_distributed_data)
        return paths

    # @timeit('get')
    def get(self, start=None, end=None, last_hours=None, column=None, with_summary=False):
        """
        Loads catalog data from disk.

        :param start: :str: or :pd.Timestamp: start datetime of data query
        :param end: :str: or :pd.Timestamp: end datetime of data query
        :param last_hours: :str: or :pd.Timestamp: query from 'last hours' until now
        :param column: desired key in pd.DataFrame
        :param with_summary: bool
        :return: data (opc: , data_summary)

        """
        start, end = _get_time_slice(start, end, last_hours, min_ts=self.min_ts)
        paths_idx = self._get_paths(start, end)
        if paths_idx:
            last, last_s = None, None
            # Incluir RAW:
            if (paths_idx[-1] == ST_TODAY) and ((last_hours is not None)
                                                or ((end is not None)
                                                    and (self.tree.ts_fin.max() >= pd.Timestamp(end)))):
                paths_idx = paths_idx[:-1]
                today = self._load_store(ST_TODAY, column=column)
                plus = self.process_data(self._load_store(self.raw_store))
                if column is not None and plus is not None:
                    plus = plus[column]
                if today is not None and plus is not None:
                    last = pd.DataFrame(pd.concat([today, plus]))
                elif today is not None:
                    last = today
                else:
                    last = plus
                if with_summary and last is not None:
                    last, last_s = self.process_data_summary(last)
            # if async_get and len(paths_idx) > 2:
            #     with futures.ProcessPoolExecutor(max_workers=min(4, len(paths_idx))) as executor:
            #         future_loads = {executor.submit(self._load_store, p, with_summary=with_summary, column=column): p
            #                         for p in paths_idx}
            #         extracted = [future.result() for future in futures.as_completed(future_loads)]
            # else:
            extracted = [self._load_store(p, with_summary=with_summary, column=column) for p in paths_idx]
            if with_summary and extracted:
                dfs, dfs_s = list(zip(*extracted))
                data = _concat_loaded_data(list(dfs) + [last], start, end, self.verbose)
                data_s = _concat_loaded_data(list(dfs_s) + [last_s], start, end, self.verbose)
                return data, data_s
            elif with_summary:
                return None, None
            return _concat_loaded_data(extracted + [last], start, end, self.verbose)
        if with_summary:
            return None, None
        return None

    @timeit('get_summary')
    def get_summary(self, start=None, end=None, last_hours=None):
        """
        Loads catalog summary data from disk.

        :param start: :str: or :pd.Timestamp: start datetime of data query
        :param end: :str: or :pd.Timestamp: end datetime of data query
        :param last_hours: :str: or :pd.Timestamp: query from 'last hours' until now
        :return: data_summary

        """
        paths_idx = self._get_paths(start, end, last_hours)
        if paths_idx:
            # if async_get and len(paths_idx) > 2:
            #     with futures.ProcessPoolExecutor(max_workers=min(4, len(paths_idx))) as executor:
            #         future_loads = {executor.submit(self._load_summary, p): p for p in paths_idx}
            #         extracted = [future.result() for future in futures.as_completed(future_loads)]
            # else:
            extracted = [self._load_summary(p) for p in paths_idx]
            data_s = _concat_loaded_data(extracted, start, end, self.verbose)
            return data_s
        return None

    @timeit('get_all_data', verbose=True)
    def get_all_data(self, with_summary_data=True):
        """
        Loads all data from catalog.

        :param with_summary_data: :bool: include summary data
        :return: data (opc: , data_summary)

        """
        return self.get(start=self.min_ts, with_summary=with_summary_data)

    @staticmethod
    def resample_data(data, rs_data=None, rm_data=None, use_median=False, func_agg=np.mean):

        def _median(arr):
            return 0 if arr.shape[0] == 0 else np.nanmedian(arr)

        if data is not None and not data.empty:
            if use_median:
                func_agg = _median
            if rm_data is not None:
                data = data.rolling(rm_data).apply(func_agg)
            elif rs_data is not None:
                data = data.resample(rs_data, label='left').apply(func_agg)
        return data

    @timeit('reprocess_all_data', verbose=True)
    def reprocess_all_data(self):
        if self.tree is not None:
            paths_w_summary = self.tree[(self.tree.key == self.key_summary) & self.tree.is_cat]
            for path in paths_w_summary.st:
                df = self._load_store(path)
                df_bis, df_s = self.process_data_summary(df)
                self._save_hdf([df_bis, df_s], path, [self.key_raw, self.key_summary], mode='w', **KWARGS_SAVE)
        else:
            log('No data to reprocess!', 'error', self.verbose)

    def get_path_hdf_store_binaries(self, rel_path=ST_TODAY):
        """
        Return abspath to hdf_store file

        :param rel_path: :str: relative path to hdf_store
        :return: :str: abspath

        """
        if self.tree is not None:
            subset = self.tree[self.tree.st.str.contains(rel_path)].st.values
            if len(subset) > 0:
                p = os.path.join(self.base_path, subset[0])
                return p
        return None

    # def info_catalog(self):
    #     # TODO Tabla de información del catálogo: ruta, archivo, n_rows, ts_ini, ts_fin, medidas de completitud
    #     raise NotImplementedError
    #
    # def backup(self, path_backup, compact_data=None):
    #     # TODO backup a ruta alternativa, con compresión y opción de replicado de tree o compactado (x años, o total)
    #     raise NotImplementedError
    #
    # def export(self):  # , export_to='csv'):
    #     # TODO terminar export?
    #     all_data, all_data_s = self.get_all_data(True, True)
    #     # if export_to == 'csv':
    #     all_data.to_csv(os.path.join(self.base_path, 'enerpi_all_data.csv'))
    #     all_data_s.to_csv(os.path.join(self.base_path, 'enerpi_all_data_summary.csv'))
