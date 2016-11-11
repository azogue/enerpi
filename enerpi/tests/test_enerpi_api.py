# -*- coding: utf-8 -*-
from enerpi.database import CONFIG_CATALOG
from enerpi.api import (enerpi_default_config, enerpi_receiver_generator, get_ts_last_save,
                        CONFIG, TZ, DATA_PATH, FILE_LOGGING, LOGGING_LEVEL)


def test_config():
    """
    Config Test

    """
    default_conf = enerpi_default_config()
    print(default_conf)
    print(TZ, DATA_PATH, FILE_LOGGING, LOGGING_LEVEL)
    print(CONFIG)
    print(CONFIG_CATALOG)


def test_get_last():
    """
    API get_ts_last_save test

    """
    print(get_ts_last_save())
    print(get_ts_last_save(get_last_sample=False, verbose=True, n=3))
    assert ()


def test_receiver_generator():
    """
    API enerpi_receiver_generator test.

    Generator used for get real-time values from current running ENERPI logger.
    It fails if some other local instance is using the same IP+PORT to decode the broadcast...

    """
    gen = enerpi_receiver_generator()
    print(next(gen))
    print(next(gen))
    print(next(gen))
    assert ()