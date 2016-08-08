# -*- coding: utf-8 -*-
import datetime as dt
import os
import pandas as pd
from time import time
import re
from shutil import copy as copy_file

from enerpi import DATA_PATH, HDF_STORE, INIT_LOG_MARK
from enerpi.base import log, funcs_tipo_output, timeit
from enerpi.pisampler import COL_TS, COLS_DATA


# Disk data default store
KEY = 'rms'
KEY_ANT = 'raw'

# Set CLI pandas width:
pd.set_option('display.width', 140)


def append_delta_y_consumo(data):
    data = data.copy()
    deltas = pd.Series(data.index).diff().fillna(method='bfill')
    frac_hora = deltas / pd.Timedelta(hours=1)
    data['Wh'] = data.power * frac_hora.values
    data['delta'] = deltas.values
    consumo = data['Wh'].rename('consumo_kWh').resample('1h', label='left').sum().divide(1000.)
    return data, consumo


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
        f_print('\n** CONSUMO ELÉCTRICO HORARIO (kWh):\n{}'.format(df_consumo))
        dias = df_consumo.resample('1D').sum()
        dias.frac /= 24
        f_print('\n*** CONSUMO ELÉCTRICO DIARIO (kWh):\n{}'.format(dias))


def load_data(path_st=HDF_STORE, filter_data=None, verbose=True, append_consumo=True):
    if os.path.exists(path_st):
        with pd.HDFStore(path_st, mode='r') as st:
            try:
                data = st[KEY]
            except KeyError:
                data = st[KEY_ANT]
                data.columns = COLS_DATA
        if filter_data:
            loc_data = filter_data.split('::')
            if len(loc_data) > 1:
                if len(loc_data[0]) > 0:
                    filtered = data.loc[loc_data[0]:loc_data[1]]
                else:
                    filtered = data.loc[:loc_data[1]]
            else:
                filtered = data.loc[loc_data[0]:]
            data = filtered
        if append_consumo:
            data, consumo = append_delta_y_consumo(data)
            return data, consumo
        else:
            return data
    log('HDF Store not found at "{}"'.format(path_st), 'error', verbose)
    if append_consumo:
        return None, None
    return None


def operate_hdf_database(raw_path_st, compact=False, path_backup=None, clear_database=False):
    # HDF Store Config
    path_st = _clean_store_path(raw_path_st)
    existe_st = os.path.exists(path_st)
    if not existe_st:
        log('HDF Store not found at "{}"'.format(path_st), 'warn', True)

    # Compactado de HDF Store
    if existe_st and compact:
        log('Se procede a compactar el HDF Store de "{}"'.format(path_st), 'info')
        df = load_data(path_st, append_consumo=False)
        show_info_data(df)
        temp_st = path_st + '_temp.h5'
        save_in_store(df, path_st=temp_st, verb=True)
        os.remove(path_st)
        os.rename(temp_st, path_st)

    # Backup de HDF Store
    if existe_st and path_backup is not None:
        path_bkp = _clean_store_path(path_backup)
        log('Se procede a hacer backup del HDF Store:\n "{}" --> "{}"'.format(path_st, path_bkp), 'ok')
        copy_file(path_st, path_bkp)

    # Borrado de HDF Store
    if existe_st and clear_database:
        log('Se procede a borrar el HDF Store en "{}"'.format(path_st), 'warn')
        os.remove(path_st)

    return path_st


def save_in_store(data, path_st=HDF_STORE, verb=True):
    with pd.HDFStore(path_st, mode='a', complevel=9, complib='zlib') as st:
        if type(data) is not pd.DataFrame:
            data = pd.DataFrame(data, columns=[COL_TS] + COLS_DATA).set_index(COL_TS).dropna().astype(float)
        st.append(KEY, data)
        df_tot = st[KEY]
    log('Tamaño Store: {:.1} KB, {} rows'.format(os.path.getsize(path_st) / 1000, len(df_tot)), 'debug', verb)
    return True


@timeit('get_ts_last_save')
def get_ts_last_save(path_st=HDF_STORE, get_last_sample=False, verbose=True, n=3):
    tic = time()
    try:
        ts = dt.datetime.fromtimestamp(os.path.getmtime(path_st))
        size_kb = os.path.getsize(path_st) / 1000
        if get_last_sample:
            with pd.HDFStore(path_st, mode='r') as st:
                df = st.select(KEY, start=-n)
                log('Store UPDATE: {:%c} , SIZE = {:.2f} KB. TOOK {:.3f} s'.format(ts, size_kb, time() - tic),
                    'debug', verbose)
                return df
        log('Store UPDATE: {:%c} , SIZE = {:.2f} KB. TOOK {:.3f} s'.format(ts, size_kb, time() - tic), 'debug', verbose)
        return ts
    except FileNotFoundError:
        log('ERROR: No se encuentra Store en {}'.format(path_st), 'err', True)
        return None


def delete_log_file(log_file, verbose=True):
    log('Se procede a borrar el LOG FILE en {} ...'.format(log_file), 'warn', verbose)
    os.remove(log_file)


@timeit('extract_log_file')
def extract_log_file(log_file, extract_temps=True, verbose=True):
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
    b_warn = df_log.tipo == 'WARNING'
    df_log.loc[b_warn, 'no_red'] = df_log[b_warn].msg.str.startswith('OSError: [Errno 101] Network is unreachable')
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
