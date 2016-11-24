# -*- coding: utf-8 -*-
import os
import pandas as pd
from enerpi.tests.conftest import TestCaseEnerpi
import enerpi.prettyprinting as pp


class TestEnerpiUpdateMonthCatalog(TestCaseEnerpi):

    subpath_test_files = 'test_update_month'
    raw_file = 'enerpi_data_test.h5'
    cat_check_integrity = False

    def test_catalog_update_month(self):
        """
        TEST UPDATE MONTH (problemas de memoria en RPI?)

        """
        print(os.listdir(self.DATA_PATH))
        print(self.path_default, self.default_before)
        n_samples = self.cat.tree[self.cat.tree['is_raw']]['n_rows'].sum()
        raw = pd.read_hdf(os.path.join(self.DATA_PATH, 'enerpi_data_test.h5'), 'rms')
        new_n_samples = raw.shape[0]

        pp.print_info(self.cat.tree)
        pp.print_cyan(self.cat.tree.describe())
        pp.print_red(raw.head())
        pp.print_red(raw.tail())
        pp.print_magenta(raw.describe())

        assert self.cat.tree.shape[0] == 31, "Diferent # of rows on catalog index ¿?"
        assert new_n_samples == 43620, "Diferent # of rows on RAW data ¿?"
        assert raw.index.is_unique, "Raw DATA with non-unique index! Can't be!"
        assert raw.index.is_monotonic_increasing, "Raw DATA with non-monotonic-increasing index! Can't be!"

        new_data = self.cat.update_catalog(data=raw)
        pp.print_ok(new_data)
        updated_n_samples = self.cat.tree[self.cat.tree['is_raw']]['n_rows'].sum()

        pp.print_info(self.cat.tree)
        pp.print_cyan(self.cat.tree.describe())
        lost_samples = 85070
        print(n_samples, new_n_samples, updated_n_samples, updated_n_samples - n_samples - new_n_samples)

        assert self.cat.tree.shape[0] == 4, "2 rows for DATA_2016_MONTH_10.h5 & 2 more for DATA_2016_11_DAY_01.h5"
        assert updated_n_samples == n_samples + new_n_samples - lost_samples, "Updated data with old and new samples"


if __name__ == '__main__':
    import unittest

    unittest.main()
