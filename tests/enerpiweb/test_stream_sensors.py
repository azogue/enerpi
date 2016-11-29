# -*- coding: utf-8 -*-
import json
import math
import os
from time import time, sleep
from tests.conftest import TestCaseEnerpiWebStreamer


class TestEnerpiWebServerStream(TestCaseEnerpiWebStreamer):

    def test_0_last_msg(self):
        self.endpoint_request('index', status_check=200)
        sleep(2)
        r = self.endpoint_request('api/last', mimetype_check='application/json', status_check=200)
        assert json.loads(r.data.decode())

    def test_1_streaming(self):

        def _print_next_msg(counter, iterator):
            msg = next(iterator)
            print('** STREAM MSG {}:\n{}\n --> T_{:.3f} secs'.format(counter, msg, time() - tic))

        tic = time()
        r = self.endpoint_request('api/stream/realtime', mimetype_check='text/event-stream', status_check=200)
        print('headers={}\nmimetype={}'.format(r.headers, r.mimetype))
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
        counter_received = 0
        for i in range(math.ceil(1.5 * self.stream_max_time)):
            try:
                _print_next_msg(i, it)
                counter_received += 1
            except StopIteration as e:
                print(e.__class__)
        self.assertGreater(counter_received, 1)
        assert not r.stream.closed
        r.stream.close()
        assert r.stream.closed


if __name__ == '__main__':
    import unittest

    unittest.main()
