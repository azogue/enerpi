# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
from enerpi.base import CONFIG, SENSORS, DATA_PATH, FILE_LOGGING, LOGGING_LEVEL, log
# noinspection PyUnresolvedReferences
from enerpi.database import init_catalog, get_ts_last_save, extract_log_file, delete_log_file, HDF_STORE
# noinspection PyUnresolvedReferences
from enerpi.enerpimeter import receiver_msg_generator, enerpi_raw_data, msg_to_dict
# noinspection PyUnresolvedReferences
from enerpi.iobroadcast import get_encryption_key, get_codec


def enerpi_receiver_generator(verbose=False, n_msgs=None):
    """
    Generator of broadcasted values by ENERPI Logger.

    It can be used by any machine in the same network as the ENERPI Logger. It decrypts the encrypted broadcast and
    returns a dict of vars - values.
    Used by the webserver for read & stream real-time values.

    :param verbose: :bool: Log to stdout
    :param n_msgs: :int: # of msgs to receive (unlimited by default).
    :return: :dict:
    """
    gen = receiver_msg_generator(verbose=verbose, n_msgs=n_msgs)
    count = 0
    while True:
        try:
            msg, _t1, _t2 = next(gen)
            yield msg_to_dict(msg)
            count += 1
        except StopIteration:
            log('EXIT from enerpi_receiver_generator. StopIteration in msg #{}'.format(count), 'error', verbose)
            break
    return None


def enerpi_default_config():
    """
    Default configuration for ENERPI Data Catalog, read from INI file.

    :return: :dict: parameters
    """
    conf = {'store': HDF_STORE,
            'DATA_PATH': DATA_PATH,
            'delta': SENSORS.delta_sec_data,
            'window': SENSORS.rms_roll_window_sec,
            'ts': SENSORS.ts_data_ms,
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
