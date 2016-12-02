# -*- coding: utf-8 -*-
import os
import enerpi.prettyprinting as pp
from tests.conftest import TestCaseEnerpiWebServer


class TestEnerpiWebServerData(TestCaseEnerpiWebServer):
    # Enerpi test scenario:
    subpath_test_files = 'test_context_2probes'
    cat_check_integrity = True

    def test_0_distribute_data(self):
        pp.print_green(os.listdir(self.DATA_PATH))
        print(self.cat.tree)
        pp.print_yellowb(self.cat)

    def test_1_download_files(self):
        print(self.cat.tree)
        self.endpoint_request("api/hdfstores/DATA_2016_MONTH_10.h5?as_attachment=true",
                              status_check=404, mimetype_check='text/html')
        self.endpoint_request("api/hdfstores/DATA_2016_MONTH_10.h5", status_check=404, mimetype_check='text/html')
        self.endpoint_request("api/hdfstores/DATA_2016_11_DAY_23.h5?as_attachment=true",
                              status_check=200, mimetype_check='application/octet-stream')
        self.endpoint_request("api/hdfstores/DATA_2016_11_DAY_23.h5",
                              status_check=200, mimetype_check='application/octet-stream')

    def test_2_context_info(self):
        self.show_context_info()

    def test_3_bokeh_plots(self):
        self.endpoint_request("api/stream/bokeh/last/5", mimetype_check='text/event-stream')
        self.endpoint_request("api/stream/bokeh/from/today", mimetype_check='text/event-stream')
        self.endpoint_request("api/stream/bokeh/from/2016-10-01/to/2016-10-05/?use_median=true&rs_data=2h",
                              status_check=404)
        self.endpoint_request("api/stream/bokeh/from/2016-10-01/to/2016-10-05/?rs_data=2h&kwh=true", status_check=404)


if __name__ == '__main__':
    import unittest

    unittest.main()
