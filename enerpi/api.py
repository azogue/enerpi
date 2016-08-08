# -*- coding: utf-8 -*-
from enerpi import DATA_PATH, HDF_STORE, FILE_LOGGING, LOGGING_LEVEL
from enerpi.database import operate_hdf_database, get_ts_last_save, load_data, show_info_data, extract_log_file
from enerpi.enerpimeter import DELTA_SEC_DATA, TS_DATA_MS, RMS_ROLL_WINDOW_SEC


def enerpi_default_config():
    conf = {'store': HDF_STORE,
            'DATA_PATH': DATA_PATH,
            'delta': DELTA_SEC_DATA,
            'window': RMS_ROLL_WINDOW_SEC,
            'ts': TS_DATA_MS,
            'LOGGING_LEVEL': LOGGING_LEVEL,
            'FILE_LOGGING': FILE_LOGGING}
    return conf


def enerpi_data(path_st=HDF_STORE, compact=False, path_backup=None, clear_database=False, show_info=True):
    if compact or path_backup:
        path_st = operate_hdf_database(path_st, compact=compact, path_backup=path_backup, clear_database=False)
    data = load_data(path_st, verbose=True, append_consumo=False)
    if clear_database:
        _ = operate_hdf_database(path_st, clear_database=clear_database)
    if show_info:
        show_info_data(data)
    return data


def get_last_saved_data(path_st=HDF_STORE, get_last_sample=True, verbose=True, n=10):
    return get_ts_last_save(path_st, get_last_sample=get_last_sample, verbose=verbose, n=n)


def enerpi_log(log_file=FILE_LOGGING, extract_temps=False, verbose=True):
    return extract_log_file(log_file, extract_temps=extract_temps, verbose=verbose)
