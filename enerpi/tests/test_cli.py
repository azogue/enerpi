# -*- coding: utf-8 -*-
import os
import subprocess
from enerpi.tests.conftest import TestCaseEnerpi
import enerpi.prettyprinting as pp


# TODO resolver problema de salida con CTRL+C (en RPI)
class TestEnerpiCLI(TestCaseEnerpi):

    def test_script_enerpi(self):
        """
        ENERPI CLI Testing with subprocess.check_output.

        """
        self.exec_subprocess(['enerpi', '-i'])
        self.exec_subprocess(['enerpi', '-l'])
        self.exec_subprocess(['enerpi', '--config'])
        self.exec_subprocess(['enerpi', '--demo', '-ts', '0', '-T', '.01', '--timeout', '2'], timeout=3)
        self.exec_subprocess(['enerpi', '--timeout', '3'], timeout=7)

    def test_main_cli(self):
        """
        CLI Testing patching sys.argv with unittest.mock

        """
        from enerpi.command_enerpi import enerpi_main_cli

        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '-i'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '-l'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '--config'], test_mode=True)
        argv = ['test_cli', '--demo', '-ts', '0', '-T', '.3', '--timeout', '3', '--temps']
        self.exec_func_with_sys_argv(enerpi_main_cli, argv, test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '-f', '-ts', '0', '-T', '.01'], test_mode=True)
        # self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '--timeout', '3'], test_mode=True)


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
