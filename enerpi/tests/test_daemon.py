# -*- coding: utf-8 -*-
import os
import sys
from unittest import TestCase
from unittest.mock import patch
import enerpi.prettyprinting as pp
from enerpi.tests.conftest import get_temp_catalog_for_testing


class TestDaemon(TestCase):
    path_default_datapath = ''
    before_tests = ''
    tmp_dir = None
    DATA_PATH = None
    cat = None

    @classmethod
    def setup_class(cls):
        """
        CRON Test Setup:
        Read existent jobs (for replace them if they are deleted)
        Also, copy example ENERPI files & sets common data catalog for testing.

        """
        # Prepara archivos:
        (tmp_dir, data_path, cat,
         path_default_datapath, before_tests) = get_temp_catalog_for_testing(subpath_test_files='test_context_enerpi')
        cls.tmp_dir = tmp_dir
        cls.DATA_PATH = data_path
        cls.cat = cat
        cls.path_default_datapath = path_default_datapath
        cls.before_tests = before_tests

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

    def test_daemon(self):
        """
        ENERPI DAEMON Testing.

        """
        from enerpi.enerdaemon import enerpi_daemon
        def _exec_cli_w_args(args):
            # noinspection PyUnresolvedReferences
            with patch.object(sys, 'argv', args):
                pp.print_secc('TESTING CLI with sys.argv: {}'.format(sys.argv))
                enerpi_daemon(test_mode=True)

        _exec_cli_w_args(['test_cli', 'status'])
        _exec_cli_w_args(['test_cli', 'stop'])
        # _exec_cli_w_args(['test_cli', 'start'])
        # _exec_cli_w_args(['test_cli', 'restart'])
        # _exec_cli_w_args(['test_cli', 'stop'])
