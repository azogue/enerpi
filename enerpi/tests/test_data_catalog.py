# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import pytest
import shutil
from unittest import TestCase
from enerpi.tests.conftest import get_temp_catalog_for_testing
import enerpi.prettyprinting as pp


def _0_catalog_update_month():
    """
    TEST UPDATE MONTH (problemas de memoria en RPI?)

    """
    (tmp_dir, data_path, cat,
     path_default, default_before) = get_temp_catalog_for_testing(subpath_test_files='test_update_month',
                                                                  check_integrity=False)

    print(os.listdir(data_path))
    n_samples = cat.tree[cat.tree['is_raw']]['n_rows'].sum()
    raw = pd.read_hdf(os.path.join(data_path, 'enerpi_data_test.h5'), 'rms')
    new_n_samples = raw.shape[0]

    pp.print_info(cat.tree)
    pp.print_cyan(cat.tree.describe())
    pp.print_red(raw.head())
    pp.print_red(raw.tail())
    pp.print_magenta(raw.describe())

    assert cat.tree.shape[0] == 31, "Diferent # of rows on catalog index ¿?"
    assert new_n_samples == 43620, "Diferent # of rows on RAW data ¿?"
    assert raw.index.is_unique, "Raw DATA with non-unique index! Can't be!"
    assert raw.index.is_monotonic_increasing, "Raw DATA with non-monotonic-increasing index! Can't be!"

    new_data = cat.update_catalog(data=raw)
    pp.print_ok(new_data)
    updated_n_samples = cat.tree[cat.tree['is_raw']]['n_rows'].sum()

    pp.print_info(cat.tree)
    pp.print_cyan(cat.tree.describe())
    lost_samples = 85070
    print(n_samples, new_n_samples, updated_n_samples, updated_n_samples - n_samples - new_n_samples)

    assert cat.tree.shape[0] == 4, "2 rows for DATA_2016_MONTH_10.h5 & 2 more for DATA_2016_11_DAY_01.h5"
    assert updated_n_samples == n_samples + new_n_samples - lost_samples, "Updated data with old and new samples"

    return tmp_dir, data_path, cat, path_default, default_before


@pytest.mark.incremental
class TestUpdateCatalog(TestCase):

    @classmethod
    def setup_class(cls):
        """
        Copy example ENERPI files & sets common data catalog for testing.

        """
        pd.set_option('display.width', 300)
        # Prepara archivos:

        tmp_dir, data_path, cat, path_default, default_before = _0_catalog_update_month()
        cls.tmp_dir = tmp_dir
        cls.DATA_PATH = data_path
        cls.cat = cat
        cls.path_default = path_default
        cls.default_before = default_before

    @classmethod
    def teardown_class(cls):
        """
        Cleanup of temp data on testing.

        """
        # Restablece default_datapath
        open(cls.path_default, 'w').write(cls.default_before)
        print('En tearDown, DATA_PATH:{}, listdir:\n{}'.format(cls.DATA_PATH, os.listdir(cls.DATA_PATH)))
        cls.tmp_dir.cleanup()

    def test_0_catalog_update(self):
        pp.print_magenta(self.cat.tree)

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
        # TODO PROFILE data_resample (muy lento!)
        print(self.cat)
        data = self.cat.load_store(self.cat.tree.st[0]).iloc[:100000]
        rs1 = self.cat.resample_data(data, rs_data='40min', rm_data=None, use_median=False, func_agg=np.mean)
        pp.print_green(rs1)
        rs2 = self.cat.resample_data(data, rm_data=30, use_median=True, func_agg=np.sqrt)
        pp.print_yellow(rs2)
        rs3 = self.cat.resample_data(data, rs_data='2min', use_median=True, func_agg=np.mean)
        pp.print_blue(rs3)
        rs3_2 = self.cat.resample_data(data, rm_data='2min', use_median=False, func_agg=np.mean)
        pp.print_blue(rs3_2)

    def test_7_catalog_operations(self):
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
        d3_none = self.cat.get(start=None, end=None, last_hours=10, column=SENSORS.main_column, with_summary=True)
        assert d2_none is None
        assert d3_none is None
        d4, d4_s = self.cat.get(start='2016-10-01', end='2016-10-02', column=SENSORS.main_column, with_summary=True)
        assert not d4.empty
        assert d4_s.shape[0] == 48

        d5 = self.cat.get(start='2016-10-01', end='2016-09-02', column=SENSORS.main_column)
        print(d5)
        assert d5 is None

    def test_8_catalog_index_update(self):
        from enerpi.base import BASE_PATH
        from enerpi.api import enerpi_data_catalog
        # Regen cat_file
        cat_file = os.path.join(self.DATA_PATH, self.cat.catalog_file)
        pp.print_secc('Regeneración de catalog_file: (se elimina "{}" y se crea con check_integrity=True)'
                      .format(cat_file))
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
        raw_data = self.cat.load_store(os.path.join(BASE_PATH, 'tests', 'rsc', 'test_update_month', 'enerpi_data_test.h5'))
        archived_data = self.cat.load_store('DATA_YEAR_2016/DATA_2016_MONTH_10.h5')
        assert self.cat.is_raw_data(raw_data)
        assert not self.cat.is_raw_data(archived_data)
        archived_data.index += pd.Timestamp.now() - raw_data.index[-1]
        raw_data.index += pd.Timestamp.now() - raw_data.index[-1]
        print(archived_data.index)
        print(raw_data.index)

        # Delete all hdf stores:
        shutil.rmtree(os.path.join(self.cat.base_path, 'DATA_YEAR_2016'))
        shutil.rmtree(os.path.join(self.cat.base_path, 'CURRENT_MONTH'))
        shutil.rmtree(os.path.join(self.cat.base_path, 'OLD_STORES'))
        os.remove(os.path.join(self.cat.base_path, self.cat.raw_store))
        print(os.listdir(self.cat.base_path))

        # Populate new hdf stores:
        archived_data.to_hdf(os.path.join(self.cat.base_path, 'PROCESSED_DATA_TO_BE_ARCHIVED.h5'), self.cat.key_raw)
        archived_data.to_hdf(os.path.join(self.cat.base_path, 'RAW_DATA_TO_BE_ARCHIVED.h5'), self.cat.key_raw)
        print(os.listdir(self.cat.base_path))

        # New catalog:
        new_cat_3 = enerpi_data_catalog(base_path=self.cat.base_path, check_integrity=True, verbose=True)
        pp.print_ok(new_cat_3)
        print(os.listdir(self.cat.base_path))
