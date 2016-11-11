# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
from enerpi.base import CONFIG, TZ, DATA_PATH, FILE_LOGGING, LOGGING_LEVEL
# noinspection PyUnresolvedReferences
from enerpi.database import init_catalog, get_ts_last_save, extract_log_file, HDF_STORE
from enerpi.enerpimeter import receiver_msg_generator, msg_to_dict, TS_DATA_MS, RMS_ROLL_WINDOW_SEC, DELTA_SEC_DATA


def enerpi_receiver_generator():
    """
    Generator of broadcasted values by ENERPI Logger.

    It can be used by any machine in the same network as the ENERPI Logger. It decrypts the encrypted broadcast and
    returns a dict of vars - values.
    Used by the webserver for read & stream real-time values.

    :return: :dict:
    """
    gen = receiver_msg_generator(False)
    count = 0
    while True:
        try:
            msg, _t1, _t2 = next(gen)
            yield msg_to_dict(msg)
            count += 1
        except StopIteration:
            print('StopIteration in it {}'.format(count))
            return None


def enerpi_default_config():
    """
    Default configuration for ENERPI Data Catalog, read from INI file.

    :return: :dict: parameters
    """
    conf = {'store': HDF_STORE,
            'DATA_PATH': DATA_PATH,
            'delta': DELTA_SEC_DATA,
            'window': RMS_ROLL_WINDOW_SEC,
            'ts': TS_DATA_MS,
            'LOGGING_LEVEL': LOGGING_LEVEL,
            'FILE_LOGGING': FILE_LOGGING}
    return conf


def enerpi_data_catalog(check_integrity=False, **kwargs):
    """
    Get ENERPI data catalog for access & operation.

    :param check_integrity: :bool: False by default. If true, checks integrity and generates / updates data index.
    :param kwargs: :dict:
    :return: :EnerpiCatalog:
    """
    return init_catalog(check_integrity=check_integrity, **kwargs)


# def get_last_saved_data(path_st=HDF_STORE, get_last_sample=True, verbose=True, n=10):
#     return get_ts_last_save(path_st, get_last_sample=get_last_sample, verbose=verbose, n=n)


# def enerpi_log(log_file=FILE_LOGGING, extract_temps=False, verbose=True):
#     return extract_log_file(log_file, extract_temps=extract_temps, verbose=verbose)
