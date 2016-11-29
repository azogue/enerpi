# -*- coding: utf-8 -*-
"""
Test catalog operations with empty catalog (no data)

"""
import os
import pandas as pd
import pytest
from tests.conftest import TestCaseEnerpi
import enerpi.prettyprinting as pp


@pytest.mark.incremental
class TestEnerpiProcessEmptyData(TestCaseEnerpi):

    cat_check_integrity = True

    def test_0_config(self):
        from enerpi.base import CONFIG
        print(self.tmp_dir)
        print(self.DATA_PATH)
        print(self.cat)
        print(CONFIG.sections())
        for s in ['ENERPI_DATA', 'ENERPI_WEBSERVER', 'ENERPI_SAMPLER', 'BROADCAST', 'RGBLED']:
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
        self.assertIsNone(path_none)

    def test_4_catalog_get_all_data(self):
        all_data = self.cat.get_all_data(with_summary_data=False)
        all_data_2, summary_data = self.cat.get_all_data(with_summary_data=True)
        self.assertIsNone(all_data)
        self.assertIsNone(all_data_2)
        self.assertIsNone(summary_data)

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
        self.assertIsNone(s_none)

    def test_6_catalog_operations(self):
        from enerpi.base import SENSORS
        data_empty = pd.DataFrame([])
        data_empty_p = self.cat.process_data(data_empty)
        assert data_empty_p.empty
        data_empty_s = self.cat.process_data_summary(data_empty)
        self.assertIsNone(data_empty_s)
        data_empty_s2, data_empty_s_ex = self.cat.process_data_summary_extra(data_empty)
        self.assertIsNone(data_empty_s2)
        self.assertIsNone(data_empty_s_ex)

        d1_none, d1_s_none = self.cat.get(start='2010-01-01', end='2010-03-01', with_summary=True)
        self.assertIsNone(d1_none)
        self.assertIsNone(d1_s_none)
        d2_none = self.cat.get(start=None, end=None, last_hours=10, column=SENSORS.main_column, with_summary=False)
        print(d2_none)
        print(self.cat.base_path)
        print(os.listdir(self.cat.base_path))
        d3_none, d3_s_none = self.cat.get(start=None, end=None, last_hours=10,
                                          column=SENSORS.main_column, with_summary=True)
        print(d3_none)
        self.assertIsNone(d2_none)
        self.assertIsNone(d3_none)
        self.assertIsNone(d3_s_none)
        d4, d4_s = self.cat.get(start='2016-10-01', end='2016-10-02', column=SENSORS.main_column, with_summary=True)
        self.assertIsNone(d4)
        self.assertIsNone(d4_s)

        d5 = self.cat.get(start='2016-10-01', end='2016-09-02', column=SENSORS.main_column)
        self.assertIsNone(d5)

    def test_7_export_data(self):
        exported_1 = self.cat.export_chunk(filename='enerpi_all_data_test_1.csv')
        pp.print_cyan(exported_1)
        self.assertEqual(exported_1, False)

    def test_8_empty_tiles(self):
        from enerpi.base import IMG_TILES_BASEPATH
        from enerpiplot.enerplot import gen_svg_tiles

        ok = gen_svg_tiles(IMG_TILES_BASEPATH, self.cat, color=(1, 0, 0))
        self.assertEqual(ok, False)


if __name__ == '__main__':
    import unittest

    unittest.main()
