# -*- coding: utf-8 -*-
"""
ENERPI - python API for accessing handy objects:
- Remote replication or data extraction
- Local configuration variables
- Local ENERPI data catalog
- Constant & config derived definitions
- Receiver value generator
...

"""
import datetime as dt
import os
import pandas as pd
import requests
import tempfile
# noinspection PyUnresolvedReferences
from enerpi.base import (ENCODING, CONFIG, SENSORS, DATA_PATH, INDEX_DATA_CATALOG, check_resource_files,
                         FILE_LOGGING, LOGGING_LEVEL, log, timeit)
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


def _request_enerpi_remote_file(url_api_download, dest_path,
                                file_mimetype='application/octet-stream', verbose=False, timeout=10):
    """
    Download remote file from ENERPI

    :param str url_api_download: download URL
    :param str dest_path: local path for remote file
    :param str file_mimetype: mimetype check
    :param bool verbose: verbose mode
    :return: download ok
    :rtype: bool

    """
    try:
        r = requests.get(url_api_download, timeout=timeout)
    except requests.exceptions.ConnectTimeout:
        log('TIMEOUT REQUEST AT "{}"'.format(url_api_download), 'error', verbose, False)
        return False
    if r.ok and (r.headers['Content-Type'] == file_mimetype):
        if verbose:
            size_kb = int(r.headers['Content-Length']) / 1024
            date_mod = pd.Timestamp(r.headers['Last-Modified']).tz_convert('Europe/Madrid')
            log('ENERPI FILE Downloaded from "{}" --> {:.2f} KB, mtime={:%d/%m/%Y %H:%M:%S}'
                .format(url_api_download, size_kb, date_mod), 'ok', verbose, False)
        check_resource_files(dest_path, verbose=False)
        with open(dest_path, 'wb') as f:
            f.write(r.content)
        if verbose:
            local_size = os.path.getsize(dest_path) / 1024
            local_date_mod = dt.datetime.fromtimestamp(os.path.getmtime(dest_path))
            log('DOWNLOADED FILE NOW IN LOCAL DISK AT "{}", {:.2f} KB, mtime={:%d/%m/%Y %H:%M:%S}'
                .format(dest_path, local_size, local_date_mod), 'magenta', verbose, False)
        return True
    log('REQUEST NOT OK TRYING TO DOWNLOAD FILE AT "{}". STATUS_CODE={}, HEADERS={}'
        .format(url_api_download, r.status_code, r.headers), 'error', verbose, False)
    return False


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
    :return: (data_key, pd.DataFrame of sliced values) pairs
    :rtype: dict

    """
    def _request_extract_enerpi_data_store(url_api_download_st):
        dest_path = os.path.join(tempfile.gettempdir(), 'temp_store.h5')
        'application/octet-stream'
        data = _request_enerpi_remote_file(url_api_download_st, dest_path, file_mimetype='application/octet-stream')
        if data:
            # Return content & remove hdf temporal file store:
            with pd.HDFStore(dest_path, 'r') as st:
                data = {k: st[k] for k in st.keys()}
                log('HDF Store downloaded:\n{}'.format(st), 'ok', verbose, False)
            os.remove(dest_path)
            return data
        return None

    paths = get_catalog_paths(t0, tf)
    url_mask = 'http://{}:{}{}/api/hdfstores/'.format(enerpi_ip, port, prefix_remote_enerpi) + '{}'
    data_stores = []
    for p in paths:
        url = url_mask.format(os.path.split(p)[-1])
        log('REQUEST HDF STORE AT: {}'.format(url), 'info', verbose, False)
        data_i = _request_extract_enerpi_data_store(url)
        if data_i is not None:
            data_stores.append(data_i)
    keys = set([k for d in data_stores for k in d.keys()])
    data_out = {k: pd.DataFrame(pd.concat([data[k].loc[t0:tf] for data in data_stores
                                           if k in data.keys()])).sort_index()
                for k in keys}
    return data_out


@timeit('replicate_remote_enerpi_data_catalog', verbose=True)
def replicate_remote_enerpi_data_catalog(local_path=DATA_PATH, enerpi_ip='192.168.1.52', port=80,
                                         prefix_remote_enerpi='/enerpi', verbose=True):
    """
    Replicate the ENERPI data catalog from a remote machine with enerpiweb running.

    :param str local_path: Local path where to replicate remote data catalog
    :param str enerpi_ip: IP of the remote machine
    :param int port: PORT of the remote enerpiweb server (Default 80)
    :param prefix_remote_enerpi: URL prefix of the remote enerpiweb server (Default '/enerpi')
    :param bool verbose: verbose mode

    """
    url_mask = 'http://{}:{}{}/api/'.format(enerpi_ip, port, prefix_remote_enerpi) + '{}'
    # Get remote index:
    csv_url = url_mask.format('filedownload/catalog')
    path_csv = os.path.join(local_path, INDEX_DATA_CATALOG)
    mimetype_csv = 'text/csv; charset={}'.format(ENCODING.lower())
    ok_catalog = _request_enerpi_remote_file(csv_url, path_csv, file_mimetype=mimetype_csv, verbose=verbose, timeout=60)
    if not ok_catalog:
        log('ERROR RETRIEVING REMOTE CATALOG FILE IN: {}, ok={}'.format(csv_url, ok_catalog), 'error', verbose, False)
        return False
    # Load retrieved catalog:
    remote_cat = enerpi_data_catalog(check_integrity=False, base_path=local_path)
    if (remote_cat.tree is None) or remote_cat.tree.empty:
        log('EMPTY REMOTE CATALOG! NOTHING TO REPLICATE. EXITING...', 'error', verbose, False)
        return False
    else:
        log('REMOTE CATALOG TO REPLICATE:\n{}'.format(remote_cat.tree), 'debug', verbose, False)
        df_stores = remote_cat.tree[remote_cat.tree.is_cat & remote_cat.tree.is_raw]
        result = {}
        ts_init = df_stores.ts_ini.min()
        for _, row in df_stores.iterrows():
            rel_path_remote_st = row.st
            log('* Replicating store "{}", with {} raw samples from {:%-d/%m/%Y} to {:%-d/%m/%Y}'
                .format(rel_path_remote_st, row.n_rows, row.ts_ini, row.ts_fin), 'debug', verbose, False)
            abs_path_new_st = os.path.join(local_path, rel_path_remote_st)
            file_id = os.path.split(rel_path_remote_st)[-1]
            url_remote_st = url_mask.format('hdfstores/' + file_id)
            ok_store_i = _request_enerpi_remote_file(url_remote_st, abs_path_new_st,
                                                     file_mimetype='application/octet-stream', verbose=verbose)
            if ok_store_i:
                result[file_id] = (int(os.path.getsize(abs_path_new_st) / 1024), row.n_rows)
            else:
                result[file_id] = (None, row.n_rows)
        # Download raw_data:
        url_remote_raw_st = url_mask.format('filedownload/raw_store')
        abs_path_new_raw_data = os.path.join(local_path, HDF_STORE)
        ok_store_raw = _request_enerpi_remote_file(url_remote_raw_st, abs_path_new_raw_data,
                                                   file_mimetype='application/octet-stream', verbose=verbose)
        # Operation report
        msg = '\nREPLICATION FROM {} DONE. DATA SINCE: {:%c}. RESULTS:\n'.format(enerpi_ip, ts_init)
        kbytes = 0
        if ok_store_raw:
            kbytes_i = int(os.path.getsize(abs_path_new_raw_data) / 1024)
            kbytes += kbytes_i
            msg += '  -> RAW DATA: {} KB\n'.format(kbytes_i)
        for k, v in result.items():
            kbytes += v[0]
            msg += '  -> STORE "{}": {} KB; {} ROWS\n'.format(k, v[0], v[1])
        msg += 'TOTAL SIZE OF DOWNLOADED DATA: {:.2f} MB'.format(kbytes / 1024)
        log(msg, 'info', verbose, False)
        return True
