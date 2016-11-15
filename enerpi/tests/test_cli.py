# -*- coding: utf-8 -*-
import os
import subprocess
from unittest import TestCase
from unittest.mock import patch
import sys
from enerpi.tests.conftest import get_temp_catalog_for_testing
import enerpi.prettyprinting as pp


class TestCLI(TestCase):

    path_default_datapath = ''
    before_tests = ''
    tmp_dir = None
    DATA_PATH = None
    cat = None

    @classmethod
    def setup_class(cls):
        """
        Copy example ENERPI files & sets common data catalog for testing.

        """
        # Prepara archivos:
        (tmp_dir, data_path, cat,
         path_default_datapath, before_tests) = get_temp_catalog_for_testing(subpath_test_files='test_context_enerpi',
                                                                             check_integrity=True)
        open(os.path.join(data_path, '.enerpi_test_key'), 'wb').write(b'AAnLKyZ_1bRBvizbBI2DRjIIY30G3DYCRY0LDWTzTsQ=')
        cls.tmp_dir = tmp_dir
        cls.DATA_PATH = data_path
        cls.cat = cat
        cls.path_default_datapath = path_default_datapath
        cls.before_tests = before_tests

    @classmethod
    def teardown_class(cls):
        """
        Cleanup of temp data on testing.

        """
        # Restablece default_datapath
        open(cls.path_default_datapath, 'w').write(cls.before_tests)
        print('En tearDown, DATA_PATH:{}, listdir:\n{}'.format(cls.DATA_PATH, os.listdir(cls.DATA_PATH)))
        cls.tmp_dir.cleanup()
        print(cls.path_default_datapath, cls.before_tests)
        print(open(cls.path_default_datapath).read())

    def test_script_enerpi(self):
        """
        ENERPI CLI Testing with subprocess.check_output.

        """
        def _exec_cli_w_args(args, timeout=None):
            try:
                out = subprocess.check_output(args, timeout=timeout).decode()
                pp.print_ok(out)
            except subprocess.TimeoutExpired as e:
                pp.print_warn(e)

        _exec_cli_w_args(['enerpi', '-i'])
        _exec_cli_w_args(['enerpi', '-l'])
        _exec_cli_w_args(['enerpi', '--config'])
        _exec_cli_w_args(['enerpi', '--demo', '-ts', '0', '-T', '.01', '--timeout', '2'], timeout=3)
        _exec_cli_w_args(['enerpi', '--timeout', '3'], timeout=7)

    def test_main_cli(self):
        """
        CLI Testing patching sys.argv with unittest.mock

        """
        from enerpi.command_enerpi import enerpi_main_cli

        def _exec_cli_w_args(args):
            # noinspection PyUnresolvedReferences
            with patch.object(sys, 'argv', args):
                pp.print_secc('TESTING CLI with sys.argv: {}'.format(sys.argv))
                enerpi_main_cli(test_mode=True)

        _exec_cli_w_args(['test_cli', '-i'])
        _exec_cli_w_args(['test_cli', '-l'])
        _exec_cli_w_args(['test_cli', '--config'])
        #  enerpi --demo -ts 1 -T .5 --timeout 15 -w .5
        _exec_cli_w_args(['test_cli', '--demo', '-ts', '0', '-T', '.3', '--timeout', '3', '--temps'])
        # _exec_cli_w_args(['test_cli', '--raw', '-ts', '0', '-T', '.3', '--timeout', '3', '--temps'])
        # _exec_cli_w_args(['test_cli', '--timeout', '3'])
        _exec_cli_w_args(['test_cli', '-f', '-ts', '0', '-T', '.01'])


def test_pitemps():
    """
    PITEMPS testing with script & module import.

    """
    from enerpi.pitemps import get_cpu_temp, get_gpu_temp

    out = subprocess.check_output('pitemps', timeout=5)
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
