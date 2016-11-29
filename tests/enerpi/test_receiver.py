# -*- coding: utf-8 -*-
"""
Test ENERPI CLI receiver (with ENERPI emitter in demo mode generating values)

"""
from tests.conftest import TestCaseEnerpiDemoStreamer


class TestEnerpiCLIReceiver(TestCaseEnerpiDemoStreamer):

    def test_main_cli_receiver(self):
        """
        CLI Testing patching sys.argv with unittest.mock

        """
        from enerpi.command_enerpi import enerpi_main_cli

        self.exec_func_with_sys_argv(enerpi_main_cli,
                                     ['test_cli', '--timeout', str(int(self.stream_max_time / 2))], test_mode=True)


if __name__ == '__main__':
    import unittest

    unittest.main()
