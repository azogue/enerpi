# -*- coding: utf-8 -*-
"""
Test ENERPI in daemon mode
"""
from tests.conftest import TestCaseEnerpi


class TestEnerpiDaemon(TestCaseEnerpi):

    def test_daemon(self):
        """
        ENERPI DAEMON Testing.

        """
        from enerpi.enerdaemon import enerpi_daemon

        self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'status'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'stop'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'bad_command'], test_mode=True)

        # Test start/restart daemon --> io.UnsupportedOperation: fileno
        # self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'start'], test_mode=True)
        # self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'restart'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'stop'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'status'], test_mode=True)


if __name__ == '__main__':
    import unittest

    unittest.main()
