# -*- coding: utf-8 -*-
import math
import os
from time import time
import unittest
from enerpi.tests.conftest import get_temp_catalog_for_testing


TIME_STREAM = 5


class TestEnerpiWebServerStream(unittest.TestCase):
    path_default_datapath = ''
    before_tests = ''
    tmp_dir = None
    DATA_PATH = None
    cat = None
    cron_orig = None
    cmd_daemon = ''

    @classmethod
    def setup_class(cls):
        # Prepara archivos:
        (tmp_dir, data_path, cat, path_default_datapath, before_tests
         ) = get_temp_catalog_for_testing(subpath_test_files='test_context_2probes', check_integrity=True)
        cls.tmp_dir = tmp_dir
        cls.DATA_PATH = data_path
        cls.cat = cat
        cls.path_default_datapath = path_default_datapath
        cls.before_tests = before_tests

        from enerpiweb import app
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.stream_max_time = TIME_STREAM
        app.config['STREAM_MAX_TIME'] = cls.stream_max_time
        cls.app = app.test_client()
        cls.app.testing = True
        print(app.config)

    @classmethod
    def teardown_class(cls):
        # Restablece default_datapath
        open(cls.path_default_datapath, 'w').write(cls.before_tests)
        print('En tearDown, DATA_PATH:{}, listdir:\n{}'.format(cls.DATA_PATH, os.listdir(cls.DATA_PATH)))
        cls.tmp_dir.cleanup()
        print(cls.path_default_datapath, cls.before_tests)
        print(open(cls.path_default_datapath).read())
        # cls.app = None

    def test_streaming(self):

        def _print_next_msg(counter, iterator):
            msg = next(iterator)
            print('** STREAM MSG {}:\n{}\n --> T_{:.3f} secs'.format(counter, msg, time() - tic))

        tic = time()
        r = self.app.get("/enerpi/api/stream/realtime")
        print('headers={}\nmimetype={}'.format(r.headers, r.mimetype))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, 'text/event-stream')
        assert r.is_streamed
        assert not r.stream.closed
        toc_get = time()
        print('GET took {:.4f} s'.format(toc_get - tic))

        print('En STREAMING, DATA_PATH:{}, listdir:\n{}'.format(self.DATA_PATH, os.listdir(self.DATA_PATH)))
        from enerpi.base import CONFIG
        key_file_now = os.path.join(self.DATA_PATH, CONFIG.get('BROADCAST', 'KEY_FILE', fallback='.secret_key'))
        print('KEY_FILE_now: {}'.format(key_file_now))
        try:
            print('KEY_now: {}'.format(open(key_file_now).read()))
        except FileNotFoundError as e:
            # Problema de .secret_key ya importada!
            print(e)

        it = r.stream.response.iter_encoded()
        for i in range(math.ceil(self.stream_max_time) - 1):
            _print_next_msg(i, it)
        try:
            _print_next_msg(-100, it)
            _print_next_msg(-200, it)
            _print_next_msg(-300, it)
            _print_next_msg(-400, it)
            assert 0
        except StopIteration as e:
            print(e.__class__)
        assert not r.stream.closed
        r.stream.close()
        assert r.stream.closed


if __name__ == '__main__':
    unittest.main()
