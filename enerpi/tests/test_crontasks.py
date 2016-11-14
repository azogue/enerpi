# # -*- coding: utf-8 -*-
from crontab import CronTab
from unittest import TestCase
from unittest.mock import patch
import sys
from enerpi.config.crontasks import set_command_on_reboot, set_command_periodic, clear_cron_commands, info_crontable
from enerpi.command_enerpi import enerpi_main_cli, make_cron_command_task_daemon


# TODO test en set_command_periodic


class TestCron(TestCase):

    cron_orig = None
    cmd_daemon = ''

    @classmethod
    def setUpClass(cls):
        """
        CRON Test Setup:
        Read existent jobs (for replace them if they are deleted)

        """
        cls.cron_orig = CronTab(user=True)
        print(cls.cron_orig.crons)
        cls.cmd_daemon = make_cron_command_task_daemon()
        print(cls.cmd_daemon)
        info_crontable()

    @classmethod
    def tearDownClass(cls):
        """
        CRON Test TearDown:
        Replace deleted jobs in the test

        """
        print(cls.cron_orig)
        post_test_cron = CronTab(user=True)
        if (cls.cmd_daemon is not None) and (cls.cmd_daemon in [c.command for c in cls.cron_orig.crons]):
            set_command_on_reboot(cls.cmd_daemon, verbose=True)
            post_test_cron = CronTab(user=True)
        assert len(cls.cron_orig.crons) == len(post_test_cron.crons)
        print([c1.command == c2.command for c1, c2 in zip(cls.cron_orig.crons, post_test_cron.crons)])
        print(cls.cron_orig.crons)
        print(post_test_cron.crons)
        assert all([c1.command == c2.command for c1, c2 in zip(cls.cron_orig.crons, post_test_cron.crons)])
        info_crontable()

    def test_cli_install_uninstall(self):
        """
        CRON + CLI Test

        """
        def _exec_cli_w_args(args):
            # noinspection PyUnresolvedReferences
            with patch.object(sys, 'argv', args):
                print('TESTING CLI with sys.argv: {}'.format(sys.argv))
                enerpi_main_cli(test_mode=True)

        print('ENERPI CLI Install & Uninstall CRON daemon: "{}"'.format(self.cmd_daemon))
        _exec_cli_w_args(['test_cli', '--install'])
        _exec_cli_w_args(['test_cli', '--uninstall'])

    def test_install_daemon(self):
        """
        CRON Test

        """
        print('Installing CRON command: "{}"'.format(self.cmd_daemon))
        set_command_on_reboot(self.cmd_daemon, verbose=True)
        set_command_on_reboot(self.cmd_daemon, comment='ENERPI DAEMON!', verbose=True)

    def test_uninstall_daemon(self):
        """
        CRON Test

        """
        print('Uninstalling CRON command: "{}"'.format(self.cmd_daemon))
        clear_cron_commands([self.cmd_daemon], verbose=True)
        clear_cron_commands([self.cmd_daemon, self.cmd_daemon], verbose=True)
