# # -*- coding: utf-8 -*-
from crontab import CronTab
import os
from unittest import TestCase
from unittest.mock import patch
import sys
from enerpi.config.crontasks import clear_cron_commands, info_crontable, set_command_periodic
from enerpi.tests.conftest import get_temp_catalog_for_testing


class TestCronWebServer(TestCase):
    path_default_datapath = ''
    before_tests = ''
    tmp_dir = None
    DATA_PATH = None
    cat = None
    cron_orig = None
    cmd_daemon = ''

    @classmethod
    def setup_class(cls):
        """
        CRON Test Setup:
        Read existent jobs (for replace them if they are deleted)
        Also, copy example ENERPI files & sets common data catalog for testing.

        """
        # Prepara archivos:
        (tmp_dir, data_path, cat, path_default_datapath, before_tests
         ) = get_temp_catalog_for_testing(subpath_test_files='test_context_2probes', check_integrity=True)
        cls.tmp_dir = tmp_dir
        cls.DATA_PATH = data_path
        cls.cat = cat
        cls.path_default_datapath = path_default_datapath
        cls.before_tests = before_tests
        cls.cron_orig = CronTab(user=True)
        print(cls.cron_orig.crons)

        from enerpiweb.command_enerpiweb import make_cron_command_task_periodic_rscgen

        cls.cmd_daemon = make_cron_command_task_periodic_rscgen()
        print(cls.cmd_daemon)
        info_crontable()

    @classmethod
    def tearDownClass(cls):
        """
        CRON Test TearDown:
        Replace deleted jobs in the test

        """
        # Restablece default_datapath
        open(cls.path_default_datapath, 'w').write(cls.before_tests)
        print('En tearDown, DATA_PATH:{}, listdir:\n{}'.format(cls.DATA_PATH, os.listdir(cls.DATA_PATH)))
        cls.tmp_dir.cleanup()
        print(cls.path_default_datapath, cls.before_tests)
        print(open(cls.path_default_datapath).read())

        print(cls.cron_orig)
        post_test_cron = CronTab(user=True)
        if (cls.cmd_daemon is not None) and (cls.cmd_daemon in [c.command for c in cls.cron_orig.crons]):
            print('(cls.cmd_daemon is not None) and (cls.cmd_daemon in [c.command for c in cls.cron_orig.crons])!!!')
            print(cls.cmd_daemon, [c.command for c in cls.cron_orig.crons])
            assert 0
            # set_command_periodic(command, comment=None, hour=None, minute=None, frecuency_days=None, verbose=True)
            # set_command_periodic(cls.cmd_daemon, verbose=True)
            # post_test_cron = CronTab(user=True)
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
        from enerpiweb.command_enerpiweb import main

        def _exec_cli_w_args(args):
            # noinspection PyUnresolvedReferences
            with patch.object(sys, 'argv', args):
                print('TESTING CLI with sys.argv: {}'.format(sys.argv))
                main()

        print('ENERPI CLI Install & Uninstall CRON daemon: "{}"'.format(self.cmd_daemon))
        _exec_cli_w_args(['test_cli_enerpiweb', '--info'])
        _exec_cli_w_args(['test_cli_enerpiweb', '--install'])
        _exec_cli_w_args(['test_cli_enerpiweb', '--info'])
        _exec_cli_w_args(['test_cli_enerpiweb', '--uninstall'])

        # with patch.object(sys, 'argv', ['test_cli_enerpiweb', '--port', '55555']):
        #     print('TESTING CLI with sys.argv: {}'.format(sys.argv))
        #     main(test_mode=True)
        # assert 0

    def test_install_daemon_rsc_gen(self):
        """
        CRON Test

        """
        print('Installing CRON command: "{}"'.format(self.cmd_daemon))
        set_command_periodic(self.cmd_daemon, verbose=True, comment='testing crontasks 3h', hour=3)
        set_command_periodic(self.cmd_daemon, verbose=True, comment='testing crontasks 30min', minute=30)

    def test_uninstall_daemon_rsc_gen(self):
        """
        CRON Test

        """
        print('Uninstalling CRON command: "{}"'.format(self.cmd_daemon))
        clear_cron_commands([self.cmd_daemon], verbose=True)
        clear_cron_commands([self.cmd_daemon, self.cmd_daemon], verbose=True)
