#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
    Simple script for RUN ENERPI TESTS one by one with coverage, and after that, combine coverage of all tests

"""
# TODO fix test context!

import datetime as dt
from glob import glob
import json
import os
import shutil
import sys
from time import time
import enerpi.prettyprinting as pp


if __name__ == '__main__':
    #
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    pp.print_secc('TESTs:')

    t1 = glob('tests/enerpi/test*.py')
    t2 = glob('tests/enerpiweb/test*.py')
    t3 = glob('tests/enerpiplot/test*.py')
    print(t1, t2, t3)
    pp.print_cyan(glob('.coverage*'))

    coverage_files = []
    test_times = []
    test_fails = []
    tic = time()
    for t in t1 + t2 + t3:
        # cmd = 'py.test --cov=enerpi --cov=enerpiplot --cov=enerpiweb -v {}'.format(t)
        cmd = 'py.test -v --cov=enerpi --cov=enerpiplot --cov=enerpiweb {}'.format(t)
        pp.print_yellowb('\n\n*** Running CMD: "{}" ***\n'.format(cmd))
        os.system(cmd)

        # Look for failed tests:
        lastfailed = json.loads(open(os.path.join('.cache', 'v', 'cache', 'lastfailed')).read())
        if lastfailed:
            pp.print_err('FAILED TESTS:\n{}'.format(lastfailed))
            [test_fails.append(k) for k in lastfailed.keys()]

        try:
            file_cov = glob('.coverage*')[0]
            temp_dest = 'temp_cov_{}_{}'.format(file_cov[1:], os.path.splitext(os.path.split(t)[-1])[0])
            shutil.move(file_cov, temp_dest)
            pp.print_magenta('** COVERAGE INFO MOVED FROM "{}" TO "{}"'.format(file_cov, temp_dest))
            coverage_files.append(temp_dest)
        except IndexError:
            pp.print_err('** NO COVERAGE DATA FROM TEST "{}". Failed info: {}'.format(t, lastfailed))

        toc = time()
        test_times.append((t, toc - tic))
        tic = toc

    print()
    pp.print_secc('COVERAGE COMBINE:')
    for fc in coverage_files:
        shutil.move(fc, fc.replace('temp_cov_', '.'))
    pp.print_ok(glob('.coverage*'))
    os.system('coverage combine')

    pp.print_secc('COVERAGE HTML:')
    os.system('coverage html')

    pp.print_ok('** TEST TIMES:')
    total = 0
    for test, took in test_times:
        pp.print_yellowb('-TEST {:40} took {:.2f} secs'.format(test, took))
        total += took
    pp.print_ok('** TOTAL TEST RUNNING TIME: {:.1f} secs for {} tests'.format(total, len(test_times)))

    if sys.platform == 'darwin':
        bkp_file = 'coverage_macos_{:%Y_%m_%d__%H%M%S}'.format(dt.datetime.now())
        shutil.copy('.coverage', bkp_file)
    else:
        bkp_file = 'coverage_rpi_{:%Y_%m_%d__%H%M%S}'.format(dt.datetime.now())
        shutil.copy('.coverage', bkp_file)
    pp.print_cyan('Total coverage file copied to "{}"'.format(bkp_file))
    if test_fails:
        pp.print_err('FAILED TESTS:')
        [pp.print_red(k) for k in test_fails]
    else:
        pp.print_ok('ALL TEST PASSED: CONGRATULATIONS!')
