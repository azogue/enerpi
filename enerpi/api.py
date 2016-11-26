# -*- coding: utf-8 -*-
import datetime as dt
import os
import pandas as pd
import requests
import tempfile
# noinspection PyUnresolvedReferences
from enerpi.base import CONFIG, SENSORS, DATA_PATH, FILE_LOGGING, LOGGING_LEVEL, log, timeit
# noinspection PyUnresolvedReferences
from enerpi.database import init_catalog, get_ts_last_save, extract_log_file, delete_log_file, HDF_STORE
# noinspection PyUnresolvedReferences
from enerpi.enerpimeter import receiver_msg_generator, enerpi_raw_data, msg_to_dict
# noinspection PyUnresolvedReferences
from enerpi.iobroadcast import get_encryption_key, get_codec
from enerpi.hdftscat import get_catalog_paths


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


@timeit('remote_data_get', verbose=True)
def remote_data_get(t0, tf=None,
                    enerpi_ip='192.168.1.52', port=80,
                    prefix_remote_enerpi='/enerpi', verbose=True):
    """
    Query a ENERPI catalog in a remote machine with enerpiweb running.

    :param t0: start of slice
    :param tf:  end of slice (or None for end = now)
    :param enerpi_ip: IP of the remote machine
    :param port: PORT of the remote enerpiweb server
    :param prefix_remote_enerpi: URL prefix of the remote enerpiweb server
    :param verbose: :bool: verbose mode
    :return: dict of (data_key, pd.DataFrame of sliced values) pairs

    """
    def _request_store(url_api_download_st, dest_path=None):
        r = requests.get(url_api_download_st)
        if r.ok and (r.headers['Content-Type'] == 'application/octet-stream'):
            if verbose:
                size_kb = int(r.headers['Content-Length']) / 1024
                date_mod = pd.Timestamp(r.headers['Last-Modified']).tz_convert('Europe/Madrid')
                log('STORE Downloaded from "{}" --> {:.2f} KB, mod={:%d/%m/%y %H:%M:%S}'
                    .format(url_api_download_st, size_kb, date_mod), 'ok', True, False)
            # delete_filest = False
            if dest_path is None:
                dest_path = os.path.join(tempfile.gettempdir(), 'temp_store.h5')
                # delete_filest = True
            with open(dest_path, 'wb') as f:
                f.write(r.content)
            if verbose:
                local_size = os.path.getsize(dest_path) / 1024
                local_date_mod = dt.datetime.fromtimestamp(os.path.getmtime(dest_path))
                log('STORE NOW IN LOCAL DISK AT "{}", {:.2f} KB, mod={:%d/%m/%y %H:%M:%S}'
                    .format(dest_path, local_size, local_date_mod), 'magenta', True, False)
            # if delete_filest:
            # Return content & remove hdf temporal file store:
            with pd.HDFStore(dest_path, 'r') as st:
                data = {k: st[k] for k in st.keys()}
                if verbose:
                    log('HDF Store downloaded:\n{}'.format(st), 'ok', True, False)
            os.remove(dest_path)
            return data

    paths = get_catalog_paths(t0, tf)
    url_mask = 'http://{}:{}{}/api/hdfstores/'.format(enerpi_ip, port, prefix_remote_enerpi) + '{}'
    data_stores = []
    for p in paths:
        url = url_mask.format(os.path.split(p)[-1])
        if verbose:
            log('REQUEST HDF STORE AT: {}'.format(url), 'info', True, False)
        data_stores.append(_request_store(url, dest_path=None))
    keys = set([k for d in data_stores for k in d.keys()])
    data_out = {k: pd.DataFrame(pd.concat([data[k].loc[t0:tf] for data in data_stores
                                           if k in data.keys()])).sort_index()
                for k in keys}
    return data_out
