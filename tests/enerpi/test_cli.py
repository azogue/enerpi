# -*- coding: utf-8 -*-
from freezegun import freeze_time
import os
# noinspection PyUnresolvedReferences
import pandas as pd
import subprocess
from tests.conftest import TestCaseEnerpi
import enerpi.prettyprinting as pp


@freeze_time("2016-10-31 23:00")
class TestEnerpiCLI(TestCaseEnerpi):

    subpath_test_files = 'test_update_month'
    cat_check_integrity = True

    def test_main_cli_info(self):
        """
        CLI Testing patching sys.argv with unittest.mock

        """
        from enerpi.command_enerpi import enerpi_main_cli

        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '-i'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '--version'], test_mode=True)

    def test_main_cli_config(self):
        """
        CLI Testing patching sys.argv with unittest.mock

        """
        from enerpi.command_enerpi import enerpi_main_cli
        from enerpi.base import DATA_PATH

        print(os.listdir(DATA_PATH))
        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '--config'], test_mode=True)

    def test_main_cli_data_filter(self):
        """
        CLI Testing patching sys.argv with unittest.mock

        """
        from enerpi.command_enerpi import enerpi_main_cli

        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '--backup', 'test_export.csv'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '--log'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '--clearlog'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '--reprocess'], test_mode=True)

    def test_main_cli_data_plots(self):
        """
        CLI Testing patching sys.argv with unittest.mock

        """
        from enerpi.command_enerpi import enerpi_main_cli
        from enerpi.base import CONFIG

        dir_plots = os.path.join(self.DATA_PATH, CONFIG.get('ENERPI_DATA', 'IMG_BASEPATH'))
        self.exec_func_with_sys_argv(enerpi_main_cli,
                                     ['test_cli', '-f', '2016-10-01::2016-10-01', '-p', 'png', '-po', 'rm=30s'],
                                     test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli,
                                     ['test_cli', '-f', '::2016-10-04', '-p', 'png', '-po', 'rm=1h'],
                                     test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli,
                                     ['test_cli', '-f', '2016-10-25::2016-10-26', '-p', 'png', '-po', 'rs=30s'],
                                     test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli,
                                     ['test_cli', '-f', '2016-10-25::2016-10-30', '-p', 'png', '-po', 'rs=3min'],
                                     test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli,
                                     ['test_cli', '-f', '2016-10-21::2016-10-31', '-p', 'png', '-po', 'rs=30min'],
                                     test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli,
                                     ['test_cli', '-f', '2016-10-10::2016-10-30', '-p', 'png', '-po', 'rs=15min'],
                                     test_mode=True)
        # To get the plots for inspection:
        # os.system('cp -R {}/ ~/Desktop/test_plots'.format(dir_plots))
        self.assertEqual(len(os.listdir(dir_plots)), 6)

    def test_main_cli_gen_tiles(self):
        """
        CLI Testing patching sys.argv with unittest.mock

        """
        from enerpi.command_enerpi import enerpi_main_cli
        from enerpi.base import IMG_TILES_BASEPATH, check_resource_files

        check_resource_files(os.path.join(IMG_TILES_BASEPATH, 'any_image'), verbose=True)
        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '--plot-tiles'], test_mode=True)

        pp.print_ok(os.listdir(IMG_TILES_BASEPATH))
        os.system('ls -la {}'.format(IMG_TILES_BASEPATH))
        # To get the tiles for inspection:
        # os.system('cp -R {}/ ~/Desktop/test_tiles'.format(IMG_TILES_BASEPATH))
        self.assertEqual(len(os.listdir(IMG_TILES_BASEPATH)), 12)


def test_pitemps():
    """
    PITEMPS testing with script & module import.

    """
    from enerpi.pitemps import get_cpu_temp, get_gpu_temp

    out = subprocess.check_output('pitemps', timeout=7)
    pp.print_ok(out)
    print(get_cpu_temp())
    print(get_gpu_temp())


def test_prettyprinting():
    """
    Pretty printing module testing.

    """
    import enerpi
    [exec('pp.{}("{}\\n{}\\nTesting prettyprinting...")'.format(f, enerpi.PRETTY_NAME, enerpi.DESCRIPTION))
     for f in dir(enerpi.prettyprinting) if f.startswith('print_')]
    d = {enerpi.BASE_PATH: os.listdir(enerpi.BASE_PATH), 'testing_pp': enerpi.PRETTY_NAME}
    pp.print_ok(pp.ppdict(d))
    pp.print_green(pp.ppdict(d, html=True))
    pp.print_green(pp.ppdict({}))
    pp.print_red(pp.ppdict(None))


if __name__ == '__main__':
    import unittest

    unittest.main()
