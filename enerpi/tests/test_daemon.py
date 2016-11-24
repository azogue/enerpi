# -*- coding: utf-8 -*-
from enerpi.tests.conftest import TestCaseEnerpi


class TestEnerpiDaemon(TestCaseEnerpi):

    def test_daemon(self):
        """
        ENERPI DAEMON Testing.

        """
        from enerpi.enerdaemon import enerpi_daemon

        self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'status'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'stop'], test_mode=True)
        # TODO Test start/restart daemon
        # self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'start'], test_mode=True)
        # self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'restart'], test_mode=True)
        # self.exec_func_with_sys_argv(enerpi_daemon, ['test_cli', 'stop'], test_mode=True)


if __name__ == '__main__':
    import unittest

    unittest.main()
