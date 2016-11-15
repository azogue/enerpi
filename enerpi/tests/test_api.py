# -*- coding: utf-8 -*-
import os
import pandas as pd
import shutil
from unittest import TestCase
from enerpi.tests.conftest import get_temp_catalog_for_testing


class TestAPI(TestCase):

    path_default_datapath = ''
    before_tests = ''
    tmp_dir = None
    DATA_PATH = None
    cat = None

    @classmethod
    def setup_class(cls):
        """
        Copy example ENERPI files & sets common data catalog for testing.

        """
        pd.set_option('display.width', 300)
        # Prepara archivos:

        tmp_dir, data_path, cat, path_default_datapath, before_tests = get_temp_catalog_for_testing()
        cls.tmp_dir = tmp_dir
        cls.DATA_PATH = data_path
        cls.cat = cat
        cls.path_default_datapath = path_default_datapath
        cls.before_tests = before_tests

    @classmethod
    def teardown_class(cls):
        """
        Cleanup of temp data on testing.

        """
        # Restablece default_datapath
        open(cls.path_default_datapath, 'w').write(cls.before_tests)
        print('En tearDown, DATA_PATH:{}, listdir:\n{}'.format(cls.DATA_PATH, os.listdir(cls.DATA_PATH)))
        cls.tmp_dir.cleanup()
        print(cls.path_default_datapath, cls.before_tests)
        print(open(cls.path_default_datapath).read())
        # assert 0

    def test_config(self):
        """
        Config Test

        """
        from enerpi.database import CONFIG_CATALOG
        from enerpi.api import enerpi_default_config, CONFIG
        default_conf = enerpi_default_config()
        print(default_conf)
        print(CONFIG)
        print(CONFIG_CATALOG)

    def test_get_last(self):
        """
        API get_ts_last_save test

        """
        from enerpi.api import get_ts_last_save, HDF_STORE
        print('HDF_STORE: ', HDF_STORE)
        from enerpi.database import HDF_STORE, HDF_STORE_PATH
        print(HDF_STORE, HDF_STORE_PATH)
        last = get_ts_last_save(path_st=os.path.join(self.cat.base_path, self.cat.raw_store),
                                get_last_sample=False, verbose=True)
        print(last, type(last))
        assert last is None
        n = 5
        df_last = get_ts_last_save(get_last_sample=True, verbose=True, n=n)
        print(df_last)
        if type(last) is pd.DataFrame:
            assert df_last.shape[0] == n
        assert get_ts_last_save(path_st='NOT_EXISTENT.h5', get_last_sample=True, verbose=True) is None

    def test_receiver_with_no_emitter(self):
        """
        API enerpi_receiver_generator test.

        """
        from enerpi.api import enerpi_receiver_generator
        gen = enerpi_receiver_generator(verbose=True, n_msgs=1)
        print(gen)
        # print(gen.gi_running)
        # print(gen.send(None))
        # print(next(gen))
        gen.close()

    def test_log_file(self):
        """
        API extract_log_file test.

        """
        from enerpi import BASE_PATH
        from enerpi.api import extract_log_file, delete_log_file
        assert not delete_log_file('NOT_EXISTENT.log', verbose=True)
        df_log_empty = extract_log_file('NOT_EXISTENT.log', extract_temps=True, verbose=True)
        assert df_log_empty.empty

        file_logging = os.path.join(BASE_PATH, 'tests', 'rsc', 'test_update_month', 'enerpi.log')
        print('file_logging: {}'.format(file_logging))
        assert os.path.exists(file_logging)
        shutil.copy(file_logging, str(self.DATA_PATH))
        new_log = os.path.join(str(self.DATA_PATH), os.path.basename(file_logging))
        df_log = extract_log_file(new_log, extract_temps=True, verbose=True)
        print(df_log)
        assert delete_log_file(new_log, verbose=True)

    def test_raw_sampling(self):
        """
        API extract_log_file test.

        """
        from enerpi.api import enerpi_raw_data
        from enerpi.base import SENSORS

        path_raw = os.path.join(self.DATA_PATH, 'test_raw_data1.h5')
        enerpi_raw_data(path_raw, roll_time=SENSORS.rms_roll_window_sec, sampling_ms=SENSORS.ts_data_ms,
                        delta_secs=4, use_dummy_sensors=True, verbose=True)

        path_raw = os.path.join(self.DATA_PATH, 'test_raw_data2.h5')
        enerpi_raw_data(path_raw, roll_time=1, sampling_ms=0, delta_secs=3, use_dummy_sensors=True, verbose=True)
