# -*- coding: utf-8 -*-
from tests.conftest import TestCaseEnerpiCRON


class TestEnerpiCRONSetup(TestCaseEnerpiCRON):

    def test_cli_install_uninstall(self):
        """
        CRON + CLI Test

        """
        from enerpi.command_enerpi import enerpi_main_cli

        print('ENERPI CLI Install & Uninstall CRON daemon: "{}"'.format(self.cmd_daemon))
        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '--install'], test_mode=True)
        self.exec_func_with_sys_argv(enerpi_main_cli, ['test_cli', '--uninstall'], test_mode=True)

    def test_install_daemon(self):
        """
        CRON Test

        """
        from enerpi.config.crontasks import set_command_on_reboot

        print('Installing CRON command: "{}"'.format(self.cmd_daemon))
        set_command_on_reboot(self.cmd_daemon, verbose=True)
        set_command_on_reboot(self.cmd_daemon, comment='ENERPI DAEMON!', verbose=True)

    def test_uninstall_daemon(self):
        """
        CRON Test

        """
        from enerpi.config.crontasks import clear_cron_commands

        print('Uninstalling CRON command: "{}"'.format(self.cmd_daemon))
        clear_cron_commands([self.cmd_daemon], verbose=True)
        clear_cron_commands([self.cmd_daemon, self.cmd_daemon], verbose=True)


if __name__ == '__main__':
    import unittest

    unittest.main()
