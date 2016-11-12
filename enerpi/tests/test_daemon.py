# -*- coding: utf-8 -*-
import sys
from unittest.mock import patch
import enerpi.prettyprinting as pp
from enerpi.enerdaemon import enerpi_daemon


def test_daemon():
    """
    ENERPI DAEMON Testing.

    """
    def _exec_cli_w_args(args):
        with patch.object(sys, 'argv', args):
            pp.print_secc('TESTING CLI with sys.argv: {}'.format(sys.argv))
            enerpi_daemon(test_mode=True)

    _exec_cli_w_args(['test_cli', 'status'])
    # _exec_cli_w_args(['test_cli', 'start'])
    _exec_cli_w_args(['test_cli', 'stop'])
    # _exec_cli_w_args(['test_cli', 'restart'])
    # _exec_cli_w_args(['test_cli', 'stop'])
