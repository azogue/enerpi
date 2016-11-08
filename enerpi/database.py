# -*- coding: utf-8 -*-
import datetime as dt
import os
import pandas as pd
from time import time
import re
from shutil import copy as copy_file
from enerpi.base import CONFIG, DATA_PATH, log, funcs_tipo_output, timeit
from enerpi.catalog import EnerpiCatalog


# Config:
INIT_LOG_MARK = CONFIG.get('ENERPI_SAMPLER', 'INIT_LOG_MARK', fallback='INIT ENERPI')
HDF_STORE = os.path.join(DATA_PATH, CONFIG.get('ENERPI_DATA', 'HDF_STORE'))

COL_TS = CONFIG.get('ENERPI_SAMPLER', 'COL_TS', fallback='ts')
COLS_DATA = CONFIG.get('ENERPI_SAMPLER', 'COLS_DATA', fallback='power, noise, ref, ldr').split(', ')
KEY = CONFIG.get('ENERPI_DATA', 'KEY', fallback='/rms')
CONFIG_CATALOG = dict(preffix='DATA',
                      raw_file=HDF_STORE,
                      key_raw_data=KEY,
                      key_summary_data='/hours',
                      key_summary_extra='/days',
                      # catalog_file=INDEX,
                      check_integrity=True,
                      archive_existent=False,
                      verbose=False,
                      backup_original=True)


def init_catalog(base_path=DATA_PATH, **kwargs):
    conf = CONFIG_CATALOG.copy()
    conf.update(base_path=base_path)
    if kwargs:
        conf.update(**kwargs)
    return EnerpiCatalog(**conf)


def _clean_store_path(path_st):
    if os.path.pathsep not in path_st:
        path_st = os.path.join(DATA_PATH, path_st)
    else:
        path_st = os.path.abspath(path_st)
    if not os.path.splitext(path_st)[1]:
        path_st += '.h5'
    return path_st


def show_info_data(df, df_consumo=None):
    f_print, _ = funcs_tipo_output('info')
    f_print('DATAFRAME INFO:\n* Head:\n{}'.format(df.head()))
    f_print('* Tail:\n{}'.format(df.tail()))
    f_print('* Count & types:\n{}'.format(pd.concat([df.count().rename('N_rows'),
                                                     df.dtypes.rename('dtypes'),
                                                     df.describe().drop('count').T], axis=1)))
    f_print, _ = funcs_tipo_output('magenta')
    if df_consumo is not None and not df_consumo.empty:
        frac_h = df.delta.resample('1h').sum().rename('frac')
        df_consumo = pd.DataFrame(pd.concat([df_consumo, (frac_h / pd.Timedelta('1h')).round(3)], axis=1))
        f_print('\n** HOURLY ELECTRICITY CONSUMPTION (kWh):\n{}'.format(df_consumo))
        dias = df_consumo.resample('1D').sum()
        dias.frac /= 24
        f_print('\n*** DAILY ELECTRICITY CONSUMPTION (kWh):\n{}'.format(dias))


# TODO Rehacer backups y clears en catalog vía CLI
def operate_hdf_database(raw_path_st, path_backup=None, clear_database=False):
    # HDF Store Config
    path_st = _clean_store_path(raw_path_st)
    existe_st = os.path.exists(path_st)
    if not existe_st:
        log('HDF Store not found at "{}"'.format(path_st), 'warn', True)

    # Backup de HDF Store
    if existe_st and path_backup is not None:
        path_bkp = _clean_store_path(path_backup)
        log('Backing up HDF Store:\n "{}" --> "{}"'.format(path_st, path_bkp), 'ok')
        copy_file(path_st, path_bkp)

    # Borrado de HDF Store
    if existe_st and clear_database:
        log('Deleting HDF Store in "{}"'.format(path_st), 'warn')
        os.remove(path_st)

    return path_st


def save_raw_data(data=None, path_st=HDF_STORE, catalog=None, verb=True):
    """
    Used in a subprocess launched from enerpimeter, this functions appends new *raw data* to the HDF raw store,
     and, if data-catalog is not None, updates it.
    :param data:
    :param path_st:
    :param catalog:
    :param verb:
    :return:
    """
    df_tot = None
    try:
        if data is not None and type(data) is not pd.DataFrame:
            data = pd.DataFrame(data, columns=[COL_TS] + COLS_DATA).set_index(COL_TS).dropna().astype(float)
            # with pd.HDFStore(path_st, mode='a') as st:
            with pd.HDFStore(path_st, mode='a', complevel=9, complib='blosc') as st:
                st.append(KEY, data)
                if catalog is not None:
                    df_tot = st[KEY]
                    log('Size Store: {:.1f} KB, {} rows'.format(os.path.getsize(path_st) / 1000, len(df_tot)),
                        'debug', verb)
            if catalog is not None:
                try:
                    catalog.update_catalog(data=df_tot)
                except Exception as e:
                    log('Exception "{}" [{}] en update_catalog (save_in_store)'
                        .format(e, e.__class__), 'error', True)
                    # TODO Notificación del error!
        return True
    except ValueError as e:
        log('ValueError en save_in_store: {}'.format(e), 'error', True)


@timeit('get_ts_last_save')
def get_ts_last_save(path_st=HDF_STORE, get_last_sample=False, verbose=True, n=3):
    tic = time()
    try:
        ts = dt.datetime.fromtimestamp(os.path.getmtime(path_st))
        size_kb = os.path.getsize(path_st) / 1000
        if get_last_sample:
            with pd.HDFStore(path_st, mode='r') as st:
                try:
                    df = st.select(KEY, start=-n)
                    log('Store UPDATE: {:%c} , SIZE = {:.2f} KB. TOOK {:.3f} s'.format(ts, size_kb, time() - tic),
                        'debug', verbose)
                    return df
                except KeyError:
                    log('ERROR: Data "{}" not found in store "{}"'.format(KEY, path_st), 'err', True)
                    return ts
        log('Store UPDATE: {:%c} , SIZE = {:.2f} KB. TOOK {:.3f} s'.format(ts, size_kb, time() - tic), 'debug', verbose)
        return ts
    except FileNotFoundError:
        log('ERROR: Store not found in {}'.format(path_st), 'err', True)
        return None


def delete_log_file(log_file, verbose=True):
    log('Deleting LOG FILE in {} ...'.format(log_file), 'warn', verbose)
    os.remove(log_file)


@timeit('extract_log_file')
def extract_log_file(log_file, extract_temps=True, verbose=True):
    if os.path.exists(log_file):
        rg_log_msg = re.compile('(?P<tipo>INFO|WARNING|DEBUG|ERROR) \[(?P<func>.+?)\] '
                                '- (?P<ts>\d{1,2}/\d\d/\d\d\d\d \d\d:\d\d:\d\d): (?P<msg>.*?)\n', re.DOTALL)
        with open(log_file, 'r') as log_f:
            df_log = pd.DataFrame(rg_log_msg.findall(log_f.read()),
                                  columns=['tipo', 'func', 'ts', 'msg'])
        df_log.drop('func', axis=1, inplace=True)
        df_log['tipo'] = df_log['tipo'].astype('category')
        df_log['ts'] = df_log['ts'].apply(lambda x: dt.datetime.strptime(x, '%d/%m/%Y %H:%M:%S'))
        df_log.loc[df_log.msg.str.startswith('Tªs --> '), 'temp'] = True
        df_log.loc[df_log.msg.str.startswith('SENDED: '), 'debug_send'] = True
        b_warn = df_log['tipo'] == 'WARNING'
        df_log.loc[b_warn, 'no_red'] = df_log[b_warn].msg.str.startswith('OSError: [Errno 101]; C_UNREACHABLE:')
        # df_log.loc[b_warn, 'no_red'] = df_log[b_warn].msg.str.startswith('OSError:  La red es inaccesible')
        df_log['exec'] = df_log['msg'].str.contains(INIT_LOG_MARK).cumsum().astype(int)
        df_log = df_log.set_index('ts')
        if extract_temps:
            rg_temps = 'Tªs --> (?P<CPU>\d{1,2}\.\d) / (?P<GPU>\d{1,2}\.\d) ºC'
            df_log = df_log.join(df_log[df_log['temp'].notnull()].msg.str.extract(rg_temps, expand=True).astype(float))
        if verbose:
            clasific = df_log.groupby(['exec', 'tipo']).count().dropna(how='all').astype(int)
            log(clasific, 'ok', True, False)
            conteo_tipos = df_log.groupby('tipo').count()
            if 'ERROR' in conteo_tipos.index:
                log(df_log[df_log.tipo == 'ERROR'].dropna(how='all', axis=1), 'error', True, False)
            if 'INFO' in conteo_tipos.index:
                log(df_log[df_log.tipo == 'INFO'].dropna(how='all', axis=1), 'info', True, False)
        return df_log
    else:
        log("extract_log_file: '{}' doesn't exists".format(log_file), 'error', verbose, False)
        return pd.DataFrame([], columns=['tipo', 'func', 'ts', 'msg'])


# if __name__ == '__main__':
#     from enerpi.api import enerpi_data_catalog
#     from enerpi.command_enerpi import set_logging_conf
#     from prettyprinting import *
#
#     set_logging_conf()
#     pd.set_option('display.width', 240)
#
#     # TEST UPDATE
#     # base = '/Users/uge/Dropbox/PYTHON/PYPROJECTS/respaldo_enerpi_rpi3/ENERPIDATA/'
#     # cat = enerpi_data_catalog(base_path=base, raw_file=os.path.join(base, 'enerpi_data.h5'),
#     # check_integrity=False, verbose=True)
#     # print_info(cat.tree)
#     #
#     # raw = pd.read_hdf(os.path.join(base, 'enerpi_data.h5'), 'rms')
#     # print_cyan(raw)
#     # print_red(raw.index.is_unique)
#     # cat.update_catalog(data=raw)
#
#     # TEST GET / GET SUMMARY
#     # Catálogo y lectura de todos los datos.
#     cat = enerpi_data_catalog()
#     cat.reprocess_all_data()
#
#     data_s = cat.get_summary(last_hours=100000)
#     data = cat.get_all_data(with_summary_data=False)
#     print_cyan(data_s)
#     print_red(data)
#     d2, data_s2 = cat.process_data_summary(data)
#     # print_info(data_s2)
#     # # print_red(data.loc['2016-08-31 23:00:00':'2016-09-01 01:00:00'])
#     common_idx = data_s.index.intersection(data_s2.index)
#     rows_iguales = (data_s.loc[common_idx].round(3).fillna(0) == data_s2.loc[common_idx].round(3).fillna(0)).all(axis=1)
#     # # print(rows_iguales)
#     print_magenta(data_s.loc[rows_iguales[~rows_iguales].index])
#     print_red(data_s2.loc[rows_iguales[~rows_iguales].index])

