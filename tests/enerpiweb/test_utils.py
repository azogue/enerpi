# -*- coding: utf-8 -*-
"""
ENERPIWEB tests:
- jinja2 filters
- CSS color operations
- CRON task for resource generation
...

"""
from tests.conftest import TestCaseEnerpiCRON, TestCaseEnerpiWeb


def test_colors():
    """
    Quick test to cover ch_color method, in enerpiplot.enerplot

    """
    from enerpiplot.enerplot import ch_color

    c1 = ch_color('aabbcc', .1)
    print(c1)
    c2 = ch_color('#aabbcc', alpha=.4)
    print(c2)
    try:
        ch_color('aabbc', alpha=.4)
        assert 0
    except ValueError:
        pass
    c3 = ch_color([.1, .5, .6], alpha=.4)
    print(c3)
    assert c3 == (0.1, 0.5, 0.6, 0.4)
    c4 = ch_color((.1, .5, .6, .8), alpha=.4)
    print(c4)
    assert c4 == (0.1, 0.5, 0.6, 0.4)
    c5 = ch_color((.1, .5, .6, .8), .5)
    print(c5)
    assert c5 == (0.05, 0.25, 0.3, 0.4)


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
        set_command_periodic(self.cmd_rscgen, verbose=True, minute=30)
        set_command_periodic(self.cmd_rscgen, verbose=True, frecuency_days=7)

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
