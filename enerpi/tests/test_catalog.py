# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import pytest
import shutil
from time import time
from enerpi.tests.conftest import TestCaseEnerpi
import enerpi.prettyprinting as pp


@pytest.mark.incremental
class TestEnerpiCatalogTasks(TestCaseEnerpi):
    subpath_test_files = 'test_update_month'
    raw_file = 'enerpi_data_test.h5'
    cat_check_integrity = False

    def test_0_catalog_update(self):
        pp.print_magenta(self.cat.tree)
        pp.print_cyan(self.cat)

    def test_1_config(self):
        from enerpi.base import DATA_PATH, CONFIG, SENSORS_THEME, SENSORS
        pp.print_yellowb(DATA_PATH)
        pp.print_yellowb(CONFIG)
        pp.print_yellowb(SENSORS_THEME)
        pp.print_yellowb(SENSORS)

    def test_2_catalog_reprocess_all_data(self):
        pp.print_magenta(self.cat.tree)
        pp.print_yellowb([os.path.basename(p) for p in os.listdir(self.DATA_PATH)])
        self.cat.reprocess_all_data()
        pp.print_yellowb(self.cat.tree)
        pp.print_cyan(os.listdir(self.DATA_PATH))
        pp.print_cyan(os.listdir(os.path.join(self.DATA_PATH, 'DATA_YEAR_2016')))
        pp.print_cyan(os.listdir(os.path.join(self.DATA_PATH, 'CURRENT_MONTH')))

    def test_3_catalog_get_paths_stores(self):
        pp.print_yellowb(self.cat.tree)
        if self.cat.tree is not None:
            for p in set(self.cat.tree['st']):
                path = self.cat.get_path_hdf_store_binaries(rel_path=p)
                pp.print_yellow('ST:{} --> PATH:{}, is_file={}'.format(p, path, os.path.isfile(path)))
                assert path is not None, 'Non existent hdf store PATH'
                assert os.path.isfile(path), 'hdf store is not a file ?'
        path_none = self.cat.get_path_hdf_store_binaries(rel_path='NO_EXISTE.h5')
        assert path_none is None

    def test_4_catalog_get_all_data(self):
        all_data = self.cat.get_all_data(with_summary_data=False)
        all_data_2, summary_data = self.cat.get_all_data(with_summary_data=True)
        assert not all_data.empty and (all_data.shape == all_data_2.shape)
        assert not summary_data.empty

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

    def test_6_catalog_data_resample(self):
        print(self.cat)
        tic = time()
        data = self.cat._load_store(self.cat.tree.st[0]).iloc[:100000]
        toc_load = time()
        rs1 = self.cat.resample_data(data, rs_data='40min', use_median=False, func_agg=np.mean)
        toc_rs40m = time()
        pp.print_green(rs1)
        rs3 = self.cat.resample_data(data, rs_data='2min', use_median=True)
        toc_rs2m_median = time()
        pp.print_blue(rs3)
        msg = 'RESAMPLE TIMES:\n'
        msg += '\t{} --> {:.3f} secs\n'.format('LOAD', toc_load - tic)
        msg += '\t{} --> {:.3f} secs\n'.format('rs40m', toc_rs40m - toc_load)
        msg += '\t{} --> {:.3f} secs\n'.format('rs2m_median', toc_rs2m_median - toc_rs40m)
        pp.print_ok(msg)

    def test_7_catalog_data_resample_11days(self):
        print(self.cat)
        tic = time()
        data = self.cat._load_store(self.cat.tree.st[0]).iloc[:1000000]
        toc_load = time()
        rs1 = self.cat.resample_data(data, rs_data='1h')
        toc_1 = time()
        pp.print_green(rs1)
        msg = 'RESAMPLE TIMES:\n'
        msg += '\t{} --> {:.3f} secs\n'.format('LOAD', toc_load - tic)
        msg += '\t{} --> {:.3f} secs\n'.format('rs_1h', toc_1 - toc_load)
        pp.print_ok(msg)

    def test_8_catalog_operations(self):
        from enerpi.base import SENSORS
        data_empty = pd.DataFrame([])
        data_empty_p = self.cat.process_data(data_empty)
        assert data_empty_p.empty
        data_empty_s = self.cat.process_data_summary(data_empty)
        assert data_empty_s is None
        data_empty_s2, data_empty_s_ex = self.cat.process_data_summary_extra(data_empty)
        assert data_empty_s2 is None
        assert data_empty_s_ex is None

        d1_none, d1_s_none = self.cat.get(start='2010-01-01', end='2010-03-01', with_summary=True)
        assert d1_none is None
        assert d1_s_none is None
        d2_none = self.cat.get(start=None, end=None, last_hours=10, column=SENSORS.main_column, with_summary=False)
        print(d2_none)
        assert (d2_none is None) or d2_none.empty
        d3_none, d3_s_none = self.cat.get(start=None, end=None, last_hours=10,
                                          column=SENSORS.main_column, with_summary=True)
        print(d3_none)
        assert d3_none is None
        assert d3_s_none is None
        d4, d4_s = self.cat.get(start='2016-10-01', end='2016-10-02', column=SENSORS.main_column, with_summary=True)
        assert not d4.empty
        assert d4_s.shape[0] == 48

        d5 = self.cat.get(start='2016-10-01', end='2016-09-02', column=SENSORS.main_column)
        print(d5)
        assert d5 is None

    def test_9_catalog_index_update(self):
        from enerpi.base import BASE_PATH
        from enerpi.api import enerpi_data_catalog
        # Regen cat_file
        cat_file = os.path.join(self.DATA_PATH, self.cat.catalog_file)
        pp.print_secc('Regeneración de catalog_file: (se elimina "{}" y se crea con check_integrity=True)'
                      .format(cat_file))
        pp.print_cyan(open(cat_file).read())
        os.remove(cat_file)
        new_cat = enerpi_data_catalog(base_path=self.cat.base_path,
                                      raw_file=self.cat.raw_store,
                                      check_integrity=True, verbose=True)
        pp.print_ok(new_cat)

        # Now with corrupted cat_file & non-existent raw_file
        with open(cat_file, 'a') as f:
            f.write('corrupting data catalog; -1\n')
        new_cat_2 = enerpi_data_catalog(base_path=self.cat.base_path,
                                        raw_file=os.path.join(self.DATA_PATH, 'enerpi_data_non_existent.h5'),
                                        check_integrity=True, verbose=True)
        pp.print_ok(new_cat_2)

        # Now with distributing data:
        raw_data = self.cat._load_store(os.path.join(BASE_PATH, 'tests', 'rsc',
                                                     'test_update_month', 'enerpi_data_test.h5'))
        archived_data = self.cat._load_store('DATA_YEAR_2016/DATA_2016_MONTH_10.h5')
        assert self.cat.is_raw_data(raw_data)
        assert not self.cat.is_raw_data(archived_data)
        archived_data.index += pd.Timestamp.now() - raw_data.index[-1]
        raw_data.index += pd.Timestamp.now() - raw_data.index[-1]
        print(archived_data.index)
        print(raw_data.index)

        # Delete all hdf stores:
        pp.print_cyan(os.listdir(self.DATA_PATH))
        pp.print_cyan(os.listdir(os.path.join(self.DATA_PATH, 'DATA_YEAR_2016')))
        pp.print_cyan(os.listdir(os.path.join(self.DATA_PATH, 'CURRENT_MONTH')))
        shutil.rmtree(os.path.join(self.cat.base_path, 'DATA_YEAR_2016'))
        shutil.rmtree(os.path.join(self.cat.base_path, 'CURRENT_MONTH'))
        shutil.rmtree(os.path.join(self.cat.base_path, 'OLD_STORES'))
        # os.remove(os.path.join(self.cat.base_path, self.cat.raw_store))

        # Populate new hdf stores:
        archived_data.to_hdf(os.path.join(self.cat.base_path, 'PROCESSED_DATA_TO_BE_ARCHIVED.h5'), self.cat.key_raw)
        archived_data.to_hdf(os.path.join(self.cat.base_path, 'RAW_DATA_TO_BE_ARCHIVED.h5'), self.cat.key_raw)
        print(os.listdir(self.cat.base_path))

        # New catalog:
        new_cat_3 = enerpi_data_catalog(base_path=self.cat.base_path, check_integrity=True, verbose=True)
        pp.print_ok(new_cat_3)
        print(os.listdir(self.cat.base_path))

    def test__10_export_data(self):
        exported_ok = self.cat.export_chunk(filename='enerpi_all_data_test_1.csv')
        pp.print_cyan(exported_ok)
        self.assertIs(exported_ok, True)


if __name__ == '__main__':
    import unittest

    unittest.main()
