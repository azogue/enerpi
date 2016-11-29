# -*- coding: utf-8 -*-
"""
Test Resource generator for ENERPIWEB (SVG tiles)

"""
from tests.conftest import TestCaseEnerpi
from freezegun import freeze_time
# noinspection PyUnresolvedReferences
import pandas as pd


@freeze_time("2016-10-31 23:00")
class TestEnerpiResourcesGenerator(TestCaseEnerpi):

    def test_tiles_generator_one_time(self):
        from enerpiplot.mule_rscgen import main

        self.exec_func_with_sys_argv(main, ['test_rscgen', '-o', '-v'], test_mode=True)

    def test_tiles_generator_loop(self):
        from enerpiplot.mule_rscgen import main

        self.exec_func_with_sys_argv(main, ['test_rscgen', '-t', '3', '-n', '2', '-v'], test_mode=True)


@freeze_time("2017-10-31 23:00")
class TestEnerpiResourcesGeneratorNoData(TestCaseEnerpi):

    def test_tiles_generator_one_time_empty(self):
        from enerpiplot.mule_rscgen import main

        self.exec_func_with_sys_argv(main, ['test_rscgen', '-o', '-v'], test_mode=True)
