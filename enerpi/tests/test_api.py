# -*- coding: utf-8 -*-
import datetime as dt
import os
import pandas as pd
import shutil
from enerpi import BASE_PATH
from enerpi.database import CONFIG_CATALOG
from enerpi.api import (enerpi_default_config, enerpi_receiver_generator, get_ts_last_save,
                        extract_log_file, delete_log_file, CONFIG)


pd.set_option('display.width', 300)


def test_config():
    """
    Config Test

    """
    default_conf = enerpi_default_config()
    print(default_conf)
    print(CONFIG)
    print(CONFIG_CATALOG)


def test_get_last():
    """
    API get_ts_last_save test

    """
    last = get_ts_last_save()
    print(last, type(last))
    assert type(last) is dt.datetime
    n = 5
    df_last = get_ts_last_save(get_last_sample=True, verbose=True, n=n)
    print(df_last)
    if type(last) is pd.DataFrame:
        assert df_last.shape[0] == n
    assert get_ts_last_save(path_st='NOT_EXISTENT.h5', get_last_sample=True, verbose=True) is None


def test_receiver_generator():
    """
    API enerpi_receiver_generator test.

    Generator used for get real-time values from current running ENERPI logger.
    It fails if some other local instance is using the same IP+PORT to decode the broadcast...

    """
    n = 3
    gen = enerpi_receiver_generator(verbose=True, n_msgs=n)
    [print(next(gen)) for _ in range(n)]
    try:
        next(gen)
    except Exception as e:
        assert e.__class__ is StopIteration


def test_log_file(tmpdir):
    """
    API extract_log_file test.

    """
    assert not delete_log_file('NOT_EXISTENT.log', verbose=True)
    df_log_empty = extract_log_file('NOT_EXISTENT.log', extract_temps=True, verbose=True)
    assert df_log_empty.empty

    file_logging = os.path.join(BASE_PATH, 'tests', 'rsc', 'test_update_month', 'enerpi.log')
    print('file_logging: {}'.format(file_logging))
    assert os.path.exists(file_logging)
    shutil.copy(file_logging, str(tmpdir))
    new_log = os.path.join(str(tmpdir), os.path.basename(file_logging))
    df_log = extract_log_file(new_log, extract_temps=True, verbose=True)
    print(df_log)
    assert delete_log_file(new_log, verbose=True)
