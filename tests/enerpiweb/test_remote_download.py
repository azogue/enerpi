# -*- coding: utf-8 -*-
"""
Test data downloading from remote running ENERPIWEB

This test needs a proper ENERPI running in another machine in the same network

"""
from tests.conftest import TestCaseEnerpi


TS_GET_START = '2016-11-28'
TS_GET_END = '2016-11-29'

BAD_IP_WITH_NO_ENERPI = '192.168.1.99'
IP_RUNNING_ENERPI = '192.168.1.44'
# IP_RUNNING_ENERPI = 'localhost'
PORT_RUNNING_ENERPI = 80
# PORT_RUNNING_ENERPI = 7777
WEBPREFIX_RUNNING_ENERPI = '/enerpi'


class TestEnerpiRemote(TestCaseEnerpi):

    def test_0_remote_replication(self):
        from enerpi.api import replicate_remote_enerpi_data_catalog

        ok = replicate_remote_enerpi_data_catalog(local_path=self.DATA_PATH,
                                                  enerpi_ip=BAD_IP_WITH_NO_ENERPI, port=PORT_RUNNING_ENERPI,
                                                  prefix_remote_enerpi=WEBPREFIX_RUNNING_ENERPI)
        print('replication OK? {}'.format(ok))
        self.assertFalse(ok, "replication OK in bad remote machine? can't be!!")

        ok = replicate_remote_enerpi_data_catalog(local_path=self.DATA_PATH,
                                                  enerpi_ip=IP_RUNNING_ENERPI, port=PORT_RUNNING_ENERPI,
                                                  prefix_remote_enerpi=WEBPREFIX_RUNNING_ENERPI)
        print('replication OK? {}'.format(ok))
        self.assertTrue(ok, 'replication NOT OK!!')

    def test_1_remote_data_get(self):
        from enerpi.api import remote_data_get

        r_data_1 = remote_data_get(TS_GET_START, tf=TS_GET_END,
                                   enerpi_ip=BAD_IP_WITH_NO_ENERPI, port=PORT_RUNNING_ENERPI,
                                   prefix_remote_enerpi=WEBPREFIX_RUNNING_ENERPI, verbose=True)
        print('r_data_1:\n', r_data_1)
        self.assertEqual(r_data_1, {}, "remote_data_get OK in bad remote machine? can't be!!")

        r_data_2 = remote_data_get(TS_GET_START, tf=TS_GET_END,
                                   enerpi_ip=IP_RUNNING_ENERPI, port=PORT_RUNNING_ENERPI,
                                   prefix_remote_enerpi=WEBPREFIX_RUNNING_ENERPI, verbose=True)
        print('r_data_2:\n', r_data_2)
        assert r_data_2

        r_data_3 = remote_data_get('2015-11-28', tf='2016-01-28',
                                   enerpi_ip=IP_RUNNING_ENERPI, port=PORT_RUNNING_ENERPI,
                                   prefix_remote_enerpi=WEBPREFIX_RUNNING_ENERPI, verbose=True)
        print('r_data_3:\n', r_data_3)


if __name__ == '__main__':
    import unittest

    unittest.main()
