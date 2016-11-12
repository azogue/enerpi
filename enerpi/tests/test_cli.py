# -*- coding: utf-8 -*-
import os
import subprocess
from unittest.mock import patch
import sys
import enerpi
import enerpi.prettyprinting as pp
from enerpi.command_enerpi import enerpi_main_cli


def test_script_enerpi():
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
    _exec_cli_w_args(['enerpi', '--demo', '-ts', '0', '-T', '.01'], timeout=3)
    _exec_cli_w_args(['enerpi', '--timeout', '5'], timeout=7)


def test_main_cli():
    """
    CLI Testing patching sys.argv with unittest.mock

    """
    def _exec_cli_w_args(args):
        with patch.object(sys, 'argv', args):
            pp.print_secc('TESTING CLI with sys.argv: {}'.format(sys.argv))
            enerpi_main_cli(test_mode=True)

    _exec_cli_w_args(['test_cli', '-i'])
    _exec_cli_w_args(['test_cli', '-l'])
    _exec_cli_w_args(['test_cli', '--config'])
    _exec_cli_w_args(['test_cli', '--demo', '-ts', '1', '-T', '.01'])
    _exec_cli_w_args(['test_cli', '--timeout', '5'])
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
    [exec('pp.{}("{}\\n{}\\nTesting prettyprinting...")'.format(f, enerpi.PRETTY_NAME, enerpi.DESCRIPTION))
     for f in dir(enerpi.prettyprinting) if f.startswith('print_')]
    d = {enerpi.BASE_PATH: os.listdir(enerpi.BASE_PATH), 'testing_pp': enerpi.PRETTY_NAME}
    pp.print_ok(pp.ppdict(d))
    pp.print_green(pp.ppdict(d, html=True))
    pp.print_green(pp.ppdict({}))
    pp.print_red(pp.ppdict(None))
