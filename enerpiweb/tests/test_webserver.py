# -*- coding: utf-8 -*-
from io import BytesIO
import json
import jsondiff
import os
import re
from enerpi.tests.conftest import TestCaseEnerpiWebServer


class TestEnerpiWebServerRoutes(TestCaseEnerpiWebServer):

    # Enerpi test scenario:
    subpath_test_files = 'test_context_2probes'
    cat_check_integrity = True

    def test_0_routes(self):

        routes_defined = {
            "api_endpoints": [
                "{}/api/filedownload/debug/".format(self.url_prefix),
                "{}/api/stream/realtime".format(self.url_prefix),
                "{}/api/stream/bokeh".format(self.url_prefix),
                "{}/api/editconfig/".format(self.url_prefix),
                "{}/api/bokehplot".format(self.url_prefix),
                "{}/api/showfile".format(self.url_prefix),
                "{}/api/monitor".format(self.url_prefix),
                "{}/api/last".format(self.url_prefix),
                "{}/api/help".format(self.url_prefix),
                "{}/control".format(self.url_prefix),
                "{}/index".format(self.url_prefix),
                "{}/".format(self.url_prefix),
                "{}/api/stream/bokeh/from/<start>/to/<end>".format(self.url_prefix),
                "{}/api/stream/bokeh/last/<last_hours>".format(self.url_prefix),
                "{}/api/stream/bokeh/from/<start>".format(self.url_prefix),
                "{}/api/filedownload/<file_id>".format(self.url_prefix),
                "{}/api/editconfig/<file>".format(self.url_prefix),
                "{}/api/uploadfile/<file>".format(self.url_prefix),
                "{}/api/hdfstores/<relpath_store>".format(self.url_prefix),
                "{}/api/showfile/<file>".format(self.url_prefix),
                "{}/api/restart/<service>".format(self.url_prefix)
            ]
        }
        result = self.endpoint_request('api/help')
        routes = json.loads(result.data.decode())
        print(routes)
        print(routes_defined)
        print(jsondiff.diff(routes, routes_defined))

        self.assertEqual(routes, routes_defined)
        # assert 0
        from enerpiweb import app
        endpoints = [rule.rule for rule in app.url_map.iter_rules() if rule.endpoint != 'static']
        self.assertEqual(routes["api_endpoints"], endpoints)
        self.assertEqual(routes_defined, json.loads(result.data.decode()))
        self.endpoint_request('notexistent', status_check=404)

    def test_1_index(self):
        self.endpoint_request('', status_check=302, verbose=True)

        self.endpoint_request("index")
        self.endpoint_request("control")
        alerta = '?alerta=%7B%22texto_alerta%22%3A+%22LOGFILE+%2FHOME%2FPI%2FENERPIDATA%2FENERPI.LOG'
        alerta += '+DELETED%22%2C+%22alert_type%22%3A+%22warning%22%7D'
        self.endpoint_request("control" + alerta)
        self.endpoint_request("api/monitor")

    def test_2_filehandler(self):
        from enerpi.editconf import ENERPI_CONFIG_FILES

        self.endpoint_request("api/editconfig/")
        self.endpoint_request("api/editconfig/flask", status_check=404)
        self.endpoint_request("api/editconfig/rsc", status_check=404)
        self.endpoint_request("api/editconfig/nginx_err", status_check=404)
        self.endpoint_request("api/editconfig/nginx", status_check=404)
        self.endpoint_request("api/editconfig/enerpi", status_check=404)
        self.endpoint_request("api/editconfig/uwsgi", status_check=404)
        self.endpoint_request("api/editconfig/raw_store", status_check=404)
        self.endpoint_request("api/editconfig/catalog", status_check=404)
        self.endpoint_request("api/editconfig/notexistent", status_check=404)

        rg_pre = re.compile('<pre>(.*)<\/pre>', flags=re.DOTALL)
        for k, checks in zip(sorted(ENERPI_CONFIG_FILES.keys()), [('[ENERPI_DATA]', 'DATA_PATH', '[BROADCAST]'),
                                                                  ('=',),
                                                                  ('analog_channel', 'is_rms', 'name')]):
            print('Config file "{}". Checking for {}'.format(k, checks))
            r = self.endpoint_request("api/editconfig/{}".format(k))
            r2 = self.endpoint_request("api/showfile/{}".format(k))
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
        self.endpoint_request("api/showfile/notexistent", status_check=404)
        # TODO tests edit configuration files + POST changes

    def test_3_download_files(self):
        from enerpi.editconf import ENERPI_CONFIG_FILES

        for file in ENERPI_CONFIG_FILES.keys():
            print('downloading id_file={}'.format(file))
            self.endpoint_request("api/filedownload/{}".format(file))
        self.endpoint_request("api/filedownload/notexistent", status_check=404)
        self.endpoint_request("api/filedownload/{}?as_attachment=true".format('config'))
        print(os.listdir(self.DATA_PATH))
        print(self.raw_file)
        self.endpoint_request("api/filedownload/{}?as_attachment=true".format('raw_store'),
                              status_check=302, verbose=True)
        self.endpoint_request("api/hdfstores/TODAY.h5", status_check=404, verbose=True)
        self.endpoint_request("api/hdfstores/TODAY.h5?as_attachment=true", status_check=404)

    def test_4_upload_files(self):
        print('test_upload_files:')
        file_bytes = BytesIO(open(os.path.join(self.DATA_PATH, 'sensors_enerpi.json'), 'rb').read())
        filename = 'other_sensors.json'
        r = self.post_file('api/uploadfile/sensors', file_bytes, filename, mimetype_check='text/html',
                           status_check=302, verbose=True)
        self.assertIn('success', r.location)
        self.assertIn('editconfig/sensors', r.location)

        file_bytes = BytesIO(open(os.path.join(self.DATA_PATH, 'config_enerpi.ini'), 'rb').read())
        filename = 'other_config.ini'
        r = self.post_file('api/uploadfile/config', file_bytes, filename, mimetype_check='text/html',
                           status_check=302, verbose=True)
        self.assertIn('success', r.location)
        self.assertIn('editconfig/config', r.location)

        file_bytes = BytesIO(open(os.path.join(self.DATA_PATH, 'secret_key_for_test'), 'rb').read())
        filename = 'secret_key'
        r = self.post_file('api/uploadfile/encryption_key', file_bytes, filename, mimetype_check='text/html',
                           status_check=302, verbose=True)
        self.assertIn('success', r.location)
        self.assertIn('editconfig/encryption_key', r.location)

        file_bytes = BytesIO(open(os.path.join(self.DATA_PATH, 'secret_key_for_test'), 'rb').read())
        self.post_file('api/uploadfile/secret_key', file_bytes, filename, status_check=500, verbose=True)
        self.endpoint_request("api/uploadfile/lala", status_check=405)

    def test_5_last_broadcast(self):
        print('LAST ENTRY:')
        self.endpoint_request("api/last", mimetype_check='application/json')

    def test_6_bokeh_plots(self):
        self.endpoint_request("api/bokehplot")
        self.endpoint_request("api/stream/bokeh", mimetype_check='text/event-stream')
        self.endpoint_request("api/stream/bokeh/last/5", mimetype_check='text/event-stream')
        self.endpoint_request("api/stream/bokeh/from/today", mimetype_check='text/event-stream')
        self.endpoint_request("api/stream/bokeh/from/2016-08-10/to/2016-08-20/?use_median=true&rs_data=2h",
                              status_check=404)
        self.endpoint_request("api/stream/bokeh/from/2016-08-01/to/2016-09-01/?rs_data=2h&kwh=true", status_check=404)
        self.endpoint_request("api/stream/bokeh/from/yesterday/to/today", mimetype_check='text/event-stream')


if __name__ == '__main__':
    import unittest

    unittest.main()
