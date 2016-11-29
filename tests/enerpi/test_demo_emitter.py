# -*- coding: utf-8 -*-
"""
Test ENERPI CLI Demo Emitter

"""
import os
from tests.conftest import TestCaseEnerpi


class TestEnerpiCLIEmitter(TestCaseEnerpi):

    def test_main_cli_demo_emitter(self):
        """
        CLI Testing patching sys.argv with unittest.mock

        """
        from enerpi.command_enerpi import enerpi_main_cli
        from enerpi.base import DATA_PATH

        print(os.listdir(DATA_PATH))
        # self.exec_func_with_sys_argv(enerpi_main_cli,
        #                              ['test_cli', '--demo', '-ts', '0', '-T', '.3', '--timeout', '5'],
        #                              test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli,
                                     ['test_cli', '--demo', '-ts', '0', '-T', '.3', '--timeout', '5', '--temps'],
                                     test_mode=True)


if __name__ == '__main__':
    import unittest

    unittest.main()
