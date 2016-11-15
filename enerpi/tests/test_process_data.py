# -*- coding: utf-8 -*-
import os
import pandas as pd
import pytest
from unittest import TestCase
import enerpi.prettyprinting as pp
from enerpi.tests.conftest import get_temp_catalog_for_testing


@pytest.mark.incremental
class TestCatalog(TestCase):

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
        pp.print_cyan('En tearDown, DATA_PATH:{}, listdir:\n{}'.format(cls.DATA_PATH, os.listdir(cls.DATA_PATH)))
        cls.tmp_dir.cleanup()
        print(cls.path_default_datapath, cls.before_tests)
        print(open(cls.path_default_datapath).read())

    def test_0_config(self):
        from enerpi.base import CONFIG
        print(self.tmp_dir)
        print(self.DATA_PATH)
        print(self.cat)
        print(CONFIG.sections())
        for s in ['ENERPI_DATA', 'ENERPI_WEBSERVER', 'ENERPI_SAMPLER', 'BROADCAST', 'MCP3008', 'RGBLED']:
            pp.print_cyan(list(CONFIG[s].keys()))
            pp.print_red(list(CONFIG[s].values()))

    def test_1_catalog_reprocess_all_data(self):
        pp.print_magenta(self.cat.tree)
        pp.print_yellowb([os.path.basename(p) for p in os.listdir(self.DATA_PATH)])
        self.cat.reprocess_all_data()

    def test_2_catalog_repr(self):
        print(self.cat)

    def test_3_catalog_get_paths_stores(self):
        if self.cat.tree is not None:
            for p in set(self.cat.tree['st']):
                path = self.cat.get_path_hdf_store_binaries(rel_path=p)
                assert path is not None, 'Non existent hdf store PATH'
                assert os.path.isfile(path), 'hdf store is not a file ?'
        path_none = self.cat.get_path_hdf_store_binaries(rel_path='NO_EXISTE.h5')
        assert path_none is None

    def test_4_catalog_get_all_data(self):
        all_data = self.cat.get_all_data(with_summary_data=False)
        all_data_2, summary_data = self.cat.get_all_data(with_summary_data=True)
        assert all_data is None
        assert all_data_2 is None
        assert summary_data is None

    def test_5_catalog_summary(self):
        pp.print_yellowb(self.cat.tree)
        t0, tf, last_hours = '2016-10-03', '2016-10-20', 1000
        s1 = self.cat.get_summary(start=t0, end=tf)
        pp.print_info(s1)
        # s2 = self.cat.get_summary(start=t0)
        # pp.print_cyan(s2)
        s3 = self.cat.get_summary(end=t0)
        pp.print_yellow(s3)
        # s4 = self.cat.get_summary(last_hours=last_hours)
        # pp.print_ok(s4)
        s_none = self.cat.get_summary(end='2010-01-01')
        assert s_none is None

    def test_6_catalog_operations(self):
        from enerpi.base import SENSORS
        data_empty = pd.DataFrame([])
        data_empty_p = self.cat.process_data(data_empty)
        assert data_empty_p.empty
        data_empty_p2, data_empty_s,  = self.cat.process_data_summary(data_empty)
        assert data_empty_p2.empty
        assert data_empty_s is None
        data_empty_p3, data_empty_s2, data_empty_s_ex = self.cat.process_data_summary_extra(data_empty)
        assert data_empty_p3.empty
        assert data_empty_s2 is None
        assert data_empty_s_ex is None

        d1_none, d1_s_none = self.cat.get(start='2010-01-01', end='2010-03-01', with_summary=True)
        assert d1_none is None
        assert d1_s_none is None
        d2_none = self.cat.get(start=None, end=None, last_hours=10, column=SENSORS.main_column, with_summary=False)
        print(d2_none)
        print(self.cat.base_path)
        print(os.listdir(self.cat.base_path))
        d3_none, d3_s_none = self.cat.get(start=None, end=None, last_hours=10,
                                          column=SENSORS.main_column, with_summary=True)
        print(d3_none)
        assert d2_none is None
        assert d3_none is None
        assert d3_s_none is None
        d4, d4_s = self.cat.get(start='2016-10-01', end='2016-10-02', column=SENSORS.main_column, with_summary=True)
        assert d4 is None
        assert d4_s is None

        d5 = self.cat.get(start='2016-10-01', end='2016-09-02', column=SENSORS.main_column)
        assert d5 is None
