# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import shutil
import tempfile
from unittest import TestCase

from enerpi.base import BASE_PATH
from enerpi.api import enerpi_data_catalog
import enerpi.prettyprinting as pp


# @pytest.mark.parametrize(argnames, argvalues): call a test function multiple times passing in different
# arguments in turn. argvalues generally needs to be a list of values if argnames specifies only one name or
# a list of tuples of values if argnames specifies multiple names. Example: @parametrize('arg1', [1,2]) would lead to
# two calls of the decorated test function, one with arg1=1 and another with arg1=2.
# see http://pytest.org/latest/parametrize.html for more info and examples.


class TestCatalog(TestCase):

    def setUp(self):
        """
        Copy example ENERPI files & sets common data catalog for testing.

        """
        print('en setUp')
        pd.set_option('display.width', 200)
        # Prepara archivos:
        files_update_month = os.path.join(BASE_PATH, 'tests', 'rsc', 'test_update_month')

        self.tmp_dir = tempfile.TemporaryDirectory(prefix='ENERPIDATA_test')
        self.DATA_PATH = self.tmp_dir.name
        # print('temp new_base: ', self.tmp_dir, '; temp DATA_PATH: ', self.DATA_PATH)

        assert os.path.exists(self.DATA_PATH)
        assert os.path.isdir(self.DATA_PATH)
        try:
            shutil.copytree(files_update_month, self.DATA_PATH)
        except FileExistsError:
            # shutil.rmtree(self.DATA_PATH)
            self.tmp_dir.cleanup()
            shutil.copytree(files_update_month, self.DATA_PATH)
        self.root_listdir = os.listdir(self.DATA_PATH)
        pp.print_red(self.root_listdir)
        self.cat = enerpi_data_catalog(base_path=self.DATA_PATH,
                                       raw_file=os.path.join(self.DATA_PATH, 'enerpi_data.h5'),
                                       check_integrity=True, verbose=True)

    def tearDown(self):
        """
        Cleanup of temp data on testing.

        """
        pp.print_cyan('En tearDown, DATA_PATH:{}, listdir:\n{}'.format(self.DATA_PATH, os.listdir(self.DATA_PATH)))
        self.tmp_dir.cleanup()

    # def test_0_catalog_update_month(self):
    #     """
    #     TEST UPDATE MONTH (problemas de memoria en RPI?)
    #
    #     """
    #     n_samples = self.cat.tree[self.cat.tree['is_raw']]['n_rows'].sum()
    #     raw = pd.read_hdf(os.path.join(self.DATA_PATH, 'enerpi_data.h5'), 'rms')
    #     new_n_samples = raw.shape[0]
    #
    #     pp.print_info(self.cat.tree.head())
    #     pp.print_info(self.cat.tree.tail())
    #     pp.print_cyan(self.cat.tree.describe())
    #     pp.print_red(raw.head())
    #     pp.print_red(raw.tail())
    #     pp.print_magenta(raw.describe())
    #
    #     assert self.cat.tree.shape[0] == 61, "Diferent # of rows on catalog index ¿?"
    #     assert new_n_samples == 43620, "Diferent # of rows on RAW data ¿?"
    #     assert raw.index.is_unique, "Raw DATA with non-unique index! Can't be!"
    #     assert raw.index.is_monotonic_increasing, "Raw DATA with non-monotonic-increasing index! Can't be!"
    #
    #     new_data = self.cat.update_catalog(data=raw)
    #     print(new_data)
    #     if new_data:
    #         self.cat = self.cat
    #     updated_n_samples = self.cat.tree[self.cat.tree['is_raw']]['n_rows'].sum()
    #
    #     pp.print_info(self.cat.tree.tail())
    #     pp.print_cyan(self.cat.tree.describe())
    #     print(n_samples, new_n_samples, updated_n_samples, updated_n_samples - n_samples - new_n_samples)
    #
    #     assert self.cat.tree.shape[0] == 4, "2 rows for DATA_2016_MONTH_10.h5 & 2 more for DATA_2016_11_DAY_01.h5"
    #     assert updated_n_samples == n_samples + new_n_samples, "Updated data with old and new saples"
    #     return False

    def test_catalog_reprocess_all_data(self):
        self.cat.reprocess_all_data()

    def test_catalog_get_paths_stores(self):
        for p in set(self.cat.tree['st']):
            path = self.cat.get_path_hdf_store_binaries(rel_path=p)
            assert path is not None, 'Non existent hdf store PATH'
            assert os.path.isfile(path), 'hdf store is not a file ?'

    def test_catalog_get_all_data(self):
        all_data = self.cat.get_all_data(with_summary_data=False)
        all_data_2, summary_data = self.cat.get_all_data(with_summary_data=True)
        assert not all_data.empty and (all_data.shape == all_data_2.shape)
        assert not summary_data.empty

    def test_catalog_repr(self):
        print(self.cat)

    def test_catalog_summary(self):
        t0, tf, last_hours = '2016-10-03', '2016-10-20', 1000
        s1 = self.cat.get_summary(start=t0, end=tf)
        pp.print_info(s1)
        s2 = self.cat.get_summary(start=t0)
        pp.print_info(s2)
        s3 = self.cat.get_summary(end=tf)
        pp.print_info(s3)
        s4 = self.cat.get_summary(last_hours=last_hours)
        pp.print_cyan(s4)
        assert ()

    def test_catalog_data_resample(self):
        print(self.cat)
        data = self.cat.load_store(self.cat.tree.st[0]).iloc[:100000]
        rs1 = self.cat.resample_data(data, rs_data='40min', rm_data=None, use_median=False, func_agg=np.mean)
        pp.print_green(rs1)
        rs2 = self.cat.resample_data(data, rm_data=30, use_median=False, func_agg=np.sqrt)
        pp.print_yellow(rs2)
        rs3 = self.cat.resample_data(data, rs_data='2min', use_median=True, func_agg=np.mean)
        pp.print_blue(rs3)
        rs3 = self.cat.resample_data(data, rm_data='2min', use_median=True, func_agg=np.mean)

