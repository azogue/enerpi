# # -*- coding: utf-8 -*-
from enerpi.tests.conftest import TestCaseEnerpiCRON, TestCaseEnerpiWeb


class TestEnerpiWebCronUtils(TestCaseEnerpiCRON):

    def test_cli_install_uninstall(self):
        """
        CRON + CLI Test

        """
        from enerpiweb.command_enerpiweb import main

        print('ENERPIWEB CLI Install & Uninstall CRON daemon: "{}"'.format(self.cmd_rscgen))
        self.exec_func_with_sys_argv(main, ['test_cli_enerpiweb', '--info'])
        self.exec_func_with_sys_argv(main, ['test_cli_enerpiweb', '--install'])
        self.exec_func_with_sys_argv(main, ['test_cli_enerpiweb', '--info'])
        self.exec_func_with_sys_argv(main, ['test_cli_enerpiweb', '--uninstall'])

    def test_install_daemon_rsc_gen(self):
        """
        CRON Test

        """
        from enerpi.config.crontasks import set_command_periodic

        print('Installing CRON command: "{}"'.format(self.cmd_rscgen))
        set_command_periodic(self.cmd_rscgen, verbose=True, comment='testing crontasks 3h', hour=3)
        set_command_periodic(self.cmd_rscgen, verbose=True, comment='testing crontasks 30min', minute=30)

    def test_uninstall_daemon_rsc_gen(self):
        """
        CRON Test

        """
        from enerpi.config.crontasks import clear_cron_commands

        print('Uninstalling CRON command: "{}"'.format(self.cmd_daemon))
        clear_cron_commands([self.cmd_rscgen], verbose=True)
        clear_cron_commands([self.cmd_rscgen, self.cmd_rscgen], verbose=True)


class TestEnerpiWebFlask(TestCaseEnerpiWeb):

    def test_jinja_filters(self):
        """
        Testing flask utils & filters

        """
        import datetime as dt
        import pandas as pd
        from flask import render_template_string
        from enerpiweb import app

        template = '''<!DOCTYPE html>
<html>
    <head>
        <title>ENERPI Web</title>
    </head>
    <body>
        <header>
        </header>
        <h1>TEST</h1>
        {% for c in colors %}
        <p>COLOR {{ c|color }}</p>
        {% endfor %}
        {% for d in dates %}
        <p>DATE {{d}} --> {{ d|text_date }}</p>
        {% endfor %}
        {% for d in date_objs %}
        <p>DATE_OBJ --> {{ d|ts_strftime }}</p>
        {% endfor %}
    </body>
</html>
        '''
        colors = [(1, 1, 0), (.875, 0.1234, 1), (125, 200, 0, .3), '#aabbcc', '123456']
        dates = ['today', 'yesterday', '1', '-3', '+7', 'error_date']
        date_objs = [dt.datetime.now(),
                     dt.datetime.now().replace(hour=0, minute=0),
                     pd.Timestamp('2016-02-29'),
                     pd.Timestamp.now(),
                     pd.Timestamp.now().replace(hour=0, minute=0),
                     'today']
        with app.app_context():
            render = render_template_string(template, colors=colors, dates=dates, date_objs=date_objs)
            print(render)


if __name__ == '__main__':
    import unittest

    unittest.main()
