# -*- coding: utf-8 -*-
# TODO TERMINAR FLASK TESTS
from io import BytesIO
import json
import jsondiff
import os
import re
from time import sleep
import unittest
from enerpi.tests.conftest import get_temp_catalog_for_testing


ROUTES = {
    "api_endpoints": [
        "/enerpi/api/stream/realtime",
        "/enerpi/api/stream/bokeh",
        "/enerpi/api/editconfig/",
        "/enerpi/api/bokehplot",
        "/enerpi/api/showfile",
        "/enerpi/api/monitor",
        "/enerpi/api/last",
        "/enerpi/api/help",
        "/enerpi/control",
        "/enerpi/index",
        "/enerpi/",
        "/enerpi/api/stream/bokeh/from/<start>/to/<end>",
        "/enerpi/api/stream/bokeh/last/<last_hours>",
        "/enerpi/api/stream/bokeh/from/<start>",
        "/enerpi/api/filedownload/<file_id>",
        "/enerpi/api/editconfig/<file>",
        "/enerpi/api/uploadfile/<file>",
        "/enerpi/api/hdfstores/<relpath_store>",
        "/enerpi/api/showfile/<file>"
    ]
}
# Todo 'content_language'


class TestEnerpiWebServer(unittest.TestCase):
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
        cls.app = app.test_client()
        cls.app.testing = True
        print(app.config)
        # Waiting to get some broadcasted values in the meantime:
        sleep(7)

    @classmethod
    def teardown_class(cls):
        # Restablece default_datapath
        open(cls.path_default_datapath, 'w').write(cls.before_tests)
        print('En tearDown, DATA_PATH:{}, listdir:\n{}'.format(cls.DATA_PATH, os.listdir(cls.DATA_PATH)))
        cls.tmp_dir.cleanup()
        print(cls.path_default_datapath, cls.before_tests)
        print(open(cls.path_default_datapath).read())
        # cls.app = None

    def _test_endpoint(self, endpoint, mimetype_check=None, status_check=200):
        print('Testing ENDPOINT={}'.format(endpoint))
        r = self.app.get(endpoint)
        self.assertEqual(r.status_code, status_check)
        print('headers={}\nmimetype={}'.format(r.headers, r.mimetype))
        if mimetype_check is not None:
            self.assertEqual(r.mimetype, mimetype_check)
        # print(r.data.decode())
        return r

    def test_routes(self):
        result = self.app.get('/enerpi/api/help')
        self.assertEqual(result.status_code, 200)
        routes = json.loads(result.data.decode())
        print(routes)
        print(ROUTES)
        print(jsondiff.diff(routes, ROUTES))

        self.assertEqual(routes, ROUTES)
        # assert 0
        from enerpiweb import app
        endpoints = [rule.rule for rule in app.url_map.iter_rules() if rule.endpoint != 'static']
        self.assertEqual(routes["api_endpoints"], endpoints)
        self.assertEqual(ROUTES, json.loads(result.data.decode()))

        self._test_endpoint('/enerpi/notexistent', status_check=404)

        result = self.app.get("/enerpi/api/stream/bokeh/from/yesterday/to/today")
        print(result.status_code)
        print(result.data.decode())
        # self.assertEqual(result.status_code, 500)

    def test_index(self):
        result_idx_1 = self.app.get("/enerpi/")
        print(result_idx_1.data.decode())
        self.assertEqual(result_idx_1.status_code, 302)

        self._test_endpoint("/enerpi/index")
        self._test_endpoint("/enerpi/control")
        alerta = '?alerta=%7B%22texto_alerta%22%3A+%22LOGFILE+%2FHOME%2FPI%2FENERPIDATA%2FENERPI.LOG'
        alerta += '+DELETED%22%2C+%22alert_type%22%3A+%22warning%22%7D'
        self._test_endpoint("/enerpi/control" + alerta)
        self._test_endpoint("/enerpi/api/monitor")

    def test_filehandler(self):
        from enerpi.editconf import ENERPI_CONFIG_FILES

        self._test_endpoint("/enerpi/api/editconfig/")

        self._test_endpoint("/enerpi/api/editconfig/flask", status_check=404)
        self._test_endpoint("/enerpi/api/editconfig/rsc", status_check=404)
        self._test_endpoint("/enerpi/api/editconfig/nginx_err", status_check=404)
        self._test_endpoint("/enerpi/api/editconfig/nginx", status_check=404)
        self._test_endpoint("/enerpi/api/editconfig/enerpi", status_check=404)
        self._test_endpoint("/enerpi/api/editconfig/uwsgi", status_check=404)
        self._test_endpoint("/enerpi/api/editconfig/raw_store", status_check=404)
        self._test_endpoint("/enerpi/api/editconfig/catalog", status_check=404)
        self._test_endpoint("/enerpi/api/editconfig/notexistent", status_check=404)

        rg_pre = re.compile('<pre>(.*)<\/pre>', flags=re.DOTALL)
        for k, checks in zip(sorted(ENERPI_CONFIG_FILES.keys()), [('[ENERPI_DATA]', 'DATA_PATH', '[BROADCAST]'),
                                                                  ('=',),
                                                                  ('analog_channel', 'is_rms', 'name')]):
            print('Config file "{}". Checking for {}'.format(k, checks))
            r = self._test_endpoint("/enerpi/api/editconfig/{}".format(k))
            r2 = self._test_endpoint("/enerpi/api/showfile/{}".format(k))
            test = r.data.decode()
            test_2 = r2.data.decode()
            lookin = rg_pre.findall(test)
            lookin_2 = rg_pre.findall(test_2)
            print(lookin_2)
            if not lookin:
                print(test)
            if not lookin_2:
                print(test_2)
            for c in checks:
                self.assertIn(c, lookin[0], 'No se encuentra "{}" en "{}"'.format(c, lookin))
                self.assertIn(c, lookin_2[0], 'No se encuentra "{}" en "{}"'.format(c, lookin))
        self._test_endpoint("/enerpi/api/showfile/notexistent", status_check=404)
        # TODO tests edit configuration files + POST changes

    def test_download_files(self):
        from enerpi.editconf import ENERPI_CONFIG_FILES

        for file in ENERPI_CONFIG_FILES.keys():
            print('downloading id_file={}'.format(file))
            self._test_endpoint("/enerpi/api/filedownload/{}".format(file))
        self._test_endpoint("/enerpi/api/filedownload/notexistent", status_check=404)
        self._test_endpoint("/enerpi/api/filedownload/{}?as_attachment=true".format('config'))
        self._test_endpoint("/enerpi/api/filedownload/{}?as_attachment=true".format('raw_store'), status_check=302)
        self._test_endpoint("/enerpi/api/hdfstores/TODAY.h5", status_check=404)
        self._test_endpoint("/enerpi/api/hdfstores/TODAY.h5?as_attachment=true", status_check=404)

    def test_upload_files(self):
        print('test_upload_files:')
        r = self.app.post('/enerpi/api/uploadfile/sensors',
                          data={'file': (BytesIO(open(os.path.join(self.DATA_PATH, 'sensors_enerpi.json'),
                                                      'rb').read()),
                                         'other_sensors.json')})
        print(r.headers)
        self.assertEquals(r.status_code, 302)
        self.assertEquals(r.mimetype, 'text/html')
        self.assertIn('success', r.location)
        self.assertIn('editconfig/sensors', r.location)

        r = self.app.post('/enerpi/api/uploadfile/config',
                          data={'file': (BytesIO(open(os.path.join(self.DATA_PATH, 'config_enerpi.ini'),
                                                      'rb').read()),
                                         'other_config.ini')})
        print(r.headers)
        self.assertEquals(r.status_code, 302)
        self.assertEquals(r.mimetype, 'text/html')
        self.assertIn('success', r.location)
        self.assertIn('editconfig/config', r.location)

        r = self.app.post('/enerpi/api/uploadfile/encryption_key',
                          data={'file': (BytesIO(open(os.path.join(self.DATA_PATH, 'secret.ini'),
                                                      'rb').read()),
                                         'secret_key')})
        print(r.headers)
        self.assertEquals(r.status_code, 302)
        self.assertEquals(r.mimetype, 'text/html')
        self.assertIn('success', r.location)
        self.assertIn('editconfig/encryption_key', r.location)

        r = self.app.post('/enerpi/api/uploadfile/secret_key',
                          data={'file': (BytesIO(open(os.path.join(self.DATA_PATH, 'secret.ini'),
                                                      'rb').read()), 'secret_key')})
        self.assertEquals(r.status_code, 500)

        self._test_endpoint("/enerpi/api/uploadfile/lala", status_check=405)

    def test_last_broadcast(self):
        print('LAST ENTRY:')
        self._test_endpoint("/enerpi/api/last", mimetype_check='application/json')

    def test_bokeh_plots(self):
        self._test_endpoint("/enerpi/api/bokehplot")
        self._test_endpoint("/enerpi/api/stream/bokeh", mimetype_check='text/event-stream')
        self._test_endpoint("/enerpi/api/stream/bokeh/last/5", mimetype_check='text/event-stream')
        self._test_endpoint("/enerpi/api/stream/bokeh/from/today", mimetype_check='text/event-stream')
        self._test_endpoint("/enerpi/api/stream/bokeh/from/2016-08-10/to/2016-08-20/?use_median=true&rs_data=2h",
                            status_check=404)
        self._test_endpoint("/enerpi/api/stream/bokeh/from/2016-08-01/to/2016-09-01/?rs_data=2h&kwh=true",
                            status_check=404)
        # _test_endpoint("/enerpi/api/stream/bokeh/from/yesterday/to/today", mimetype_check='text/event-stream')


if __name__ == '__main__':
    unittest.main()
