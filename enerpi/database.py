# -*- coding: utf-8 -*-
"""
ENERPI - Database methods:

- Get/Init/Update ENERPI data catalog
- Process or clear log files
- Method for appending raw data to ENERPI catalog from the ENERPI Logger
...

"""
import datetime as dt
import os
import pandas as pd
from time import time
import re
from enerpi.base import CONFIG, DATA_PATH, log, timeit, SENSORS
from enerpi.catalog import EnerpiCatalog


# Config:
INIT_LOG_MARK = CONFIG.get('ENERPI_SAMPLER', 'INIT_LOG_MARK', fallback='INIT ENERPI')
HDF_STORE = CONFIG.get('ENERPI_DATA', 'HDF_STORE')
HDF_STORE_PATH = os.path.join(DATA_PATH, HDF_STORE)

KEY = CONFIG.get('ENERPI_DATA', 'KEY', fallback='/rms')
CONFIG_CATALOG = dict(raw_file=HDF_STORE,
                      key_raw_data=KEY,
                      key_summary_data='/hours',
                      key_summary_extra='/days',
                      check_integrity=True,
                      archive_existent=False,
                      verbose=False)


def init_catalog(sensors=None, base_path=DATA_PATH, **kwargs):
    """
    Get ENERPI data catalog for access & operation with params.

    :param sensors: Sensors config object (class EnerpiSamplerConf)
    :param base_path: :str: ENERPIDATA base path
    :param kwargs: :dict: parameters
    :return: :EnerpiCatalog:
    """
    conf = CONFIG_CATALOG.copy()
    conf.update(base_path=base_path)
    if kwargs:
        conf.update(**kwargs)
    return EnerpiCatalog(sensors=sensors, **conf)


def _clean_store_path(path_st):
    if os.path.pathsep not in path_st:
        path_st = os.path.join(DATA_PATH, path_st)
    else:
        path_st = os.path.abspath(path_st)
    if not os.path.splitext(path_st)[1]:
        path_st += '.h5'
    return path_st


def show_info_data(df, df_consumo=None):
    """
    Prints some info about DATA (& opt SUMMARY_DATA)

    :param df: :pd.DataFrame: DATA
    :param df_consumo: :pd.DataFrame: SUMMARY_DATA
    """
    log('DATAFRAME INFO:\n* Head:\n{}'.format(df.head()), 'info', True, False)
    log('* Tail:\n{}'.format(df.tail()), 'info', True, False)
    log('* Count & types:\n{}'
        .format(pd.concat([df.count().rename('N_rows'), df.dtypes.rename('dtypes'), df.describe().drop('count').T],
                          axis=1)), 'info', True, False)
    if df_consumo is not None and not df_consumo.empty:
        log('\n** HOURLY ELECTRICITY CONSUMPTION (kWh):\n{}'.format(df_consumo), 'magenta', True, False)
        dias = df_consumo.drop(['p_min', 'p_mean', 'p_max'], axis=1).resample('1D').sum()
        p_rs = df_consumo[['p_min', 'p_max']].resample('1D')
        dias = dias.join(p_rs.p_min.min()).join(p_rs.p_max.max())
        dias['t_ref'] /= 24
        log('\n*** DAILY ELECTRICITY CONSUMPTION (kWh):\n{}'.format(dias), 'ok', True, False)


def _notify_error_in_save_raw_data(msg_error):
    from time import sleep
    from enerpi.notifier import push_enerpi_error

    t = push_enerpi_error('SAVE RAW DATA', msg_error)
    sleep(1)
    return t


def save_raw_data(data=None, path_st=HDF_STORE_PATH, catalog=None, verb=True):
    """
    Used in a subprocess launched from enerpimeter, this functions appends new *raw data* to the HDF raw store,
     and, if data-catalog is not None, updates it.
    :param data:
    :param path_st:
    :param catalog:
    :param verb:
    :return:
    """
    try:
        df_tot = None
        if data is not None and type(data) is not pd.DataFrame:
            data = pd.DataFrame(data, columns=SENSORS.columns_sampling
                                ).set_index(SENSORS.ts_column).dropna().astype(float)
            mode = 'a' if os.path.exists(path_st) else 'w'
            try:
                with pd.HDFStore(path_st, mode=mode, complevel=9, complib='blosc') as st:
                    st.append(KEY, data)
                    if catalog is not None:
                        df_tot = st[KEY]
                        log('Size Store: {:.1f} KB, {} rows'.format(os.path.getsize(path_st) / 1000, len(df_tot)),
                            'debug', verb)
            except OSError as e:
                msg_error = 'OSError "{}" trying to open "{}" in "{}" mode (save_in_store)'.format(e, path_st, mode)
                log(msg_error, 'error', True)
                _notify_error_in_save_raw_data(msg_error)
                return -1
            if catalog is not None:
                try:
                    catalog.update_catalog(data=df_tot)
                except Exception as e:
                    msg_error = 'Exception "{}" [{}] en update_catalog (save_in_store)'.format(e, e.__class__)
                    log(msg_error, 'error', True)
                    _notify_error_in_save_raw_data(msg_error)
                    return -1
        return True
    except ValueError as e:
        log('ValueError en save_in_store: {}'.format(e), 'error', True)
        return -1


@timeit('get_ts_last_save')
def get_ts_last_save(path_st=HDF_STORE_PATH, get_last_sample=False, verbose=True, n=3):
    """
    Returns last data timestamp in hdf store.

    :param path_st: :str: hdf store file path
    :param get_last_sample: :bool: returns ts or pd.DataFrame
    :param verbose: :bool: shows logging msgs in stdout.
    :param n: :int: # of tail rows
    :return: pd.Timestamp or pd.DataFrame
    """
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
    """
    Removes (logging) file from disk.

    :param log_file: :str: logging file path
    :param verbose: :bool: shows logging msgs in stdout.
    """
    if os.path.exists(log_file) and os.path.isfile(log_file):
        log('Deleting LOG FILE in {} ...'.format(log_file), 'warn', verbose, False)
        os.remove(log_file)
        return True
    return False


@timeit('extract_log_file')
def extract_log_file(log_file, extract_temps=True, verbose=True):
    """
    Extracts pd.DataFrame from logging file.

    :param log_file: :str: logging file path
    :param extract_temps: :bool: process RPI temperature logging entries (appends columns 'CPU' & 'GPU')
    :param verbose: :bool: shows logging msgs in stdout.
    :return: time-indexed pd.DataFrame
    """
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
