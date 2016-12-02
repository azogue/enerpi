# -*- coding: utf-8 -*-
"""
ENERPIWEB tests - Send Email methods

"""
import os
import enerpi.prettyprinting as pp
from tests.conftest import TestCaseEnerpiWebServer


class TestEnerpiWebEmail(TestCaseEnerpiWebServer):
    # Enerpi test scenario:
    subpath_test_files = 'test_context_2probes'
    cat_check_integrity = False

    def test_0_set_context(self):
        pp.print_green(os.listdir(self.DATA_PATH))
        print(self.cat.tree)
        pp.print_yellowb(self.cat)

    def test_1_email_status(self):
        # email to default recipient:
        r = self.endpoint_request("api/email/status", status_check=302, mimetype_check='text/html')
        pp.print_ok(r.data.decode())

        # email to multiple recipients:
        r = self.endpoint_request("api/email/status/example@mail.com,eugenio.panadero@gmail.com",
                                  status_check=302, mimetype_check='text/html')
        pp.print_cyan(r.data.decode())


if __name__ == '__main__':
    import unittest

    unittest.main()
