# -*- coding: utf-8 -*-
"""
# Custom classes for testing ENERPI (unittest TestCase's subclassing)

Test Cases with the ability of reproduce different scenarios,
without changing anything in the user custom configuration nor its ENERPI_DATA.

"""
from crontab import CronTab
from glob import glob
import os
import pytest
import shutil
from subprocess import check_output, TimeoutExpired, Popen
import sys
import tempfile
from time import sleep
from unittest import TestCase
from unittest.mock import patch
from enerpi import BASE_PATH
import enerpi.prettyprinting as pp


TIME_STREAM = 5
DEBUG_EXTRA_INFO = os.path.join(BASE_PATH, '..', 'debug_tests_info.txt')


# from http://doc.pytest.org/en/latest/example/simple.html - incremental testing - test steps
def pytest_runtest_makereport(item, call):
    """
    Pytest helper for incremental tests (with decorator: @pytest.mark.incremental)

    """
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    """
    Pytest helper for incremental tests (with decorator: @pytest.mark.incremental)

    """
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed ({})".format(previousfailed.name))


def _get_temp_catalog_for_testing(subpath_test_files='test_context_enerpi',
                                  raw_file='enerpi_data_test.h5', check_integrity=True):
    """
    Copy example ENERPI files & sets common data catalog for testing.

    """
    print('TEST DEBUGGING: in get_temp_catalog_for_testing')
    dir_config = os.path.join(BASE_PATH, 'config')
    path_default_datapath = os.path.join(dir_config, '.enerpi_data_path')
    before_tests = open(path_default_datapath).read()

    # Prepara archivos:
    path_files_test = os.path.join(BASE_PATH, 'tests', 'rsc', subpath_test_files)
    tmp_dir = tempfile.TemporaryDirectory(prefix='ENERPIDATA_test')
    data_path = tmp_dir.name
    open(path_default_datapath, 'w').write(data_path)
    try:
        shutil.copytree(path_files_test, data_path)
    except FileExistsError:
        tmp_dir.cleanup()  # shutil.rmtree(data_path)
        shutil.copytree(path_files_test, data_path)
    # with patch('builtins.input', return_value='1'):
    #     from enerpi.base import reload_config
    #     from enerpi.api import enerpi_data_catalog
    #     cat = enerpi_data_catalog(base_path=data_path, raw_file=raw_file,
    # check_integrity=check_integrity, verbose=True)

    from enerpi.base import reload_config
    reload_config()
    from enerpi.api import enerpi_data_catalog
    cat = enerpi_data_catalog(base_path=data_path, raw_file=raw_file, check_integrity=check_integrity, verbose=True)

    return tmp_dir, data_path, cat, path_default_datapath, before_tests


# Test Cases for ENERPI
class TestCaseEnerpi(TestCase):

    tmp_dir = None
    DATA_PATH = None
    cat = None
    path_default = None
    default_before = None

    # Default test scenario
    subpath_test_files = 'test_context_enerpi'
    raw_file = 'enerpi_data_test.h5'
    cat_check_integrity = False

    @classmethod
    def setup_class(cls):
        """
        Copy example ENERPI files & sets common data catalog for testing.

        """

        print('SetUp (baseclass): SUBPATH_TEST_FILES: {}; cat_check_integrity={}'
              .format(cls.subpath_test_files, cls.cat_check_integrity))
        (tmp_dir, data_path, cat, path_default_datapath, before_tests
         ) = _get_temp_catalog_for_testing(subpath_test_files=cls.subpath_test_files,
                                           raw_file=cls.raw_file, check_integrity=cls.cat_check_integrity)
        cls.tmp_dir = tmp_dir
        cls.DATA_PATH = data_path
        cls.cat = cat
        cls.path_default = path_default_datapath
        cls.default_before = before_tests
        print('Finish SetUp (baseclass): TMP_DIR: {};\nDATA_PATH:{}, listdir:\n{}\n, Prev default DATA_PATH:{}'
              .format(cls.tmp_dir, cls.DATA_PATH, os.listdir(cls.DATA_PATH), cls.default_before))
        with open(DEBUG_EXTRA_INFO, 'a') as f:
            f.write('Setup TEST [{}] in "{}"\n'.format(cls.__class__, cls.DATA_PATH))
        # pp.print_red('** CONTEXT HIST:\n{}'.format(open(DEBUG_EXTRA_INFO).read()))

    @classmethod
    def teardown_class(cls):
        """
        Cleanup of temp data on testing.

        """
        # Restablece default_datapath
        print('En tearDown (baseclass): DATA_PATH:{}, listdir:\n{}\n, Prev default DATA_PATH:{}'
              .format(cls.DATA_PATH, os.listdir(cls.DATA_PATH), cls.default_before))
        open(cls.path_default, 'w').write(cls.default_before)
        shutil.rmtree(cls.tmp_dir.name)
        cls.tmp_dir.cleanup()
        with open(DEBUG_EXTRA_INFO, 'a') as f:
            f.write('Teardown TEST [{}]. Cleanup of tmp_dir at "{}"\n'.format(cls.__class__, cls.DATA_PATH))

    @classmethod
    def show_context_info(cls):
        """
        Print some info about the test context (for debug purposes)

        """
        pp.print_secc('TEXT CONTEXT INFORMATION:')
        pp.print_info('* DATA_PATH: {}\n* DEFAULT DATA_PATH: {}\n* DEFAULT BEFORE: {}\n'
                      .format(cls.DATA_PATH, cls.path_default, cls.default_before))
        pp.print_redb('* DEFAULT DATA_PATH NOW: {}'.format(open(cls.path_default).read()))
        pp.print_yellowb('** TEST SCENARIO: "{}":\n-> RAW_FILE: "{}"; cat_check_integrity={}'
                         .format(cls.subpath_test_files, cls.raw_file, cls.cat_check_integrity))
        pp.print_magenta('** CATALOG:\n{}'.format(cls.cat))
        pp.print_cyan('** FILES:\n -- {}'.format('\n -- '.join(glob('{}/**'.format(cls.DATA_PATH), recursive=True))))

    @staticmethod
    def exec_func_with_sys_argv(func_exec, custom_argv, *args_func_exec, **kwargs_func_exec):
        """
        Exec a CLI function patching sys.argv.
        For test CLI main functions with argparse

        :param func_exec:
        :param custom_argv:
        :param kwargs_func_exec:

        """
        # noinspection PyUnresolvedReferences
        with patch.object(sys, 'argv', custom_argv):
            print('TESTING CLI with sys.argv: {}'.format(sys.argv))
            func_exec(*args_func_exec, **kwargs_func_exec)

    @staticmethod
    def exec_subprocess(cmd, timeout=None):
        """
        Check output of CLI command
        :param cmd: list with splitted command
        :param timeout: optional # of seconds
        :return: output of CLI command

        """
        try:
            out = check_output(cmd, timeout=timeout).decode()
            pp.print_ok(out)
            return out
        except TimeoutExpired as e:
            pp.print_warn(e)
            return None


class TestCaseEnerpiCRON(TestCaseEnerpi):

    cron_orig = None
    cmd_daemon = None
    cmd_rscgen = None

    @classmethod
    def setup_class(cls):
        """
        CRON Test Setup:
        Read existent jobs (for replace them if they are deleted)
        Also, copy example ENERPI files & sets common data catalog for testing.

        """
        super(TestCaseEnerpiCRON, cls).setup_class()
        # Lee user CRON:
        cls.cron_orig = CronTab(user=True)
        print(cls.cron_orig.crons)

        from enerpi.command_enerpi import make_cron_command_task_daemon
        from enerpiweb.command_enerpiweb import make_cron_command_task_periodic_rscgen
        from enerpi.config.crontasks import info_crontable

        cls.cmd_daemon = make_cron_command_task_daemon()
        cls.cmd_rscgen = make_cron_command_task_periodic_rscgen()
        print(cls.cmd_daemon)
        print(cls.cmd_rscgen)
        info_crontable()

    @classmethod
    def teardown_class(cls):
        """
        CRON Test TearDown:
        Replace deleted jobs in the test

        """
        # Restablece default_datapath
        super(TestCaseEnerpiCRON, cls).teardown_class()

        from enerpi.config.crontasks import set_command_on_reboot, info_crontable, set_command_periodic

        print(cls.cron_orig)
        post_test_cron = CronTab(user=True)
        if (cls.cmd_daemon is not None) and (cls.cmd_daemon in [c.command for c in cls.cron_orig.crons]):
            set_command_on_reboot(cls.cmd_daemon, verbose=True)
            post_test_cron = CronTab(user=True)
        if (cls.cmd_rscgen is not None) and (cls.cmd_rscgen in [c.command for c in cls.cron_orig.crons]):
            set_command_periodic(cls.cmd_rscgen, comment='Generador de recursos para ENERPIWEB',
                                 minute=10, verbose=True)
            post_test_cron = CronTab(user=True)

        assert len(cls.cron_orig.crons) == len(post_test_cron.crons)
        print([c1.command == c2.command for c1, c2 in zip(cls.cron_orig.crons, post_test_cron.crons)])
        print(cls.cron_orig.crons)
        print(post_test_cron.crons)
        assert all([c1.command == c2.command for c1, c2 in zip(cls.cron_orig.crons, post_test_cron.crons)])
        info_crontable()


class TestCaseEnerpiWeb(TestCaseEnerpi):

    app = None
    stream_max_time = TIME_STREAM
    url_prefix = None

    @classmethod
    def setup_class(cls):
        """
        Copy example ENERPI files, sets common data catalog & starts webserver for testing.

        """
        super(TestCaseEnerpiWeb, cls).setup_class()

        from enerpiweb import app, PREFIX_WEB, CONFIG

        cls.url_prefix = PREFIX_WEB
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['STREAM_MAX_TIME'] = cls.stream_max_time
        cls.app = app.test_client()
        cls.app.testing = True
        print(app.config)
        pp.print_yellowb(dict(CONFIG['ENERPI_WEBSERVER']))
        pp.print_yellowb(dict(CONFIG['BROADCAST']))

    @classmethod
    def endpoint_request(cls, endpoint, mimetype_check=None, status_check=200, verbose=False):
        """
        Test GET request with the Flask app.test_client()
        :param endpoint: route to test (without app config preffix)
        :param mimetype_check: optional str for mimetype checking
        :param status_check: int for checking the request status_code
        :param verbose: bool for printing decoded request data
        :return: response object
        """
        url = '{}/{}'.format(cls.url_prefix, endpoint)
        print('Testing ENDPOINT={}, URL={}'.format(endpoint, url))
        r = cls.app.get(url)
        if verbose:
            try:
                print(r.data.decode())
            except UnicodeDecodeError as e:
                print('UnicodeDecodeError={}, trying bytes data:'.format(e))
                print(r.data)
        if r.status_code != status_check:  # Print info before the status code assert, for context inspection
            cls.show_context_info()
        assert r.status_code == status_check
        print('headers={}\nmimetype={}'.format(r.headers, r.mimetype))
        if mimetype_check is not None:
            assert r.mimetype == mimetype_check
        return r

    def post_file(self, endpoint, file_bytes, filename, mimetype_check=None, status_check=200, verbose=False):
        url = '{}/{}'.format(self.url_prefix, endpoint)
        print('upload_file "{}" with POST in URL={}'.format(filename, url))
        r = self.app.post(url, data={'file': (file_bytes, filename)})
        pp.print_cyan('HEADERS: {}'.format(r.headers))
        if verbose:
            pp.print_red(r.data)
        self.assertEquals(r.status_code, status_check)
        if mimetype_check is not None:
            self.assertEquals(r.mimetype, mimetype_check)
        return r


class TestCaseEnerpiWebStreamer(TestCaseEnerpiWeb):

    @classmethod
    def setup_class(cls):
        """
        Same as TestCaseEnerpiWeb, and starts a demo emitter for stream testing

        """
        super(TestCaseEnerpiWebStreamer, cls).setup_class()
        cls.endpoint_request('index', mimetype_check='text/html', verbose=True)

        # Starting demo emitter:
        cmd = ['enerpi', '--demo', '-ts', '3', '-T', '1', '--timeout', str(cls.stream_max_time)]
        print('Popen cmd "{}"'.format(cmd))
        Popen(cmd)


class TestCaseEnerpiWebServer(TestCaseEnerpiWeb):

    @classmethod
    def setup_class(cls):
        """
        Same as TestCaseEnerpiWeb, but sleeps some time after one first request,
        for buffering some broadcasted values in the meantime

        """
        super(TestCaseEnerpiWebServer, cls).setup_class()
        cls.endpoint_request('index', mimetype_check='text/html', verbose=True)

        # Waiting to get some broadcasted values in the meantime:
        sleep(5)
