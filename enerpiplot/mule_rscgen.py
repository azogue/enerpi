# -*- coding: utf-8 -*-
"""
ENERPIPLOT - Resource Generator for ENERPIWEB

- CLI for SVG tiles generation with last hours sensor data
- It can be used in a CRON task or like a UWSGI mule module

"""
import argparse
import os
from time import time, sleep
# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')
# noinspection PyPep8
from enerpi.base import (check_resource_files, log, set_logging_conf,
                         STATIC_PATH, SERVER_FILE_LOGGING_RSCGEN, LOGGING_LEVEL, IMG_TILES_BASEPATH)
# noinspection PyPep8
from enerpiplot.enerplot import gen_svg_tiles
# noinspection PyPep8
from enerpi.database import init_catalog


VERBOSE = False
TILES_GENERATION_INTERVAL_LOOP = 180

# Establecemos logging
LOGGING_LEVEL_SERVER = LOGGING_LEVEL
set_logging_conf(SERVER_FILE_LOGGING_RSCGEN, LOGGING_LEVEL_SERVER, with_initial_log=True, verbose=VERBOSE)

basedir_enerweb = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'enerpiweb')
check_resource_files(STATIC_PATH, os.path.join(basedir_enerweb, 'static'), verbose=VERBOSE)
check_resource_files(os.path.join(IMG_TILES_BASEPATH, 'any_image'), verbose=VERBOSE)


###############################
# RESOURCE GENERATOR
###############################
def _rsc_generator(catalog, verbose=False):
    tic, ok = time(), False
    log('**RESOURCE_GENERATOR desde MULE con PID={}!!!'.format(os.getpid()), 'debug', verbose)
    ok = gen_svg_tiles(IMG_TILES_BASEPATH, catalog, last_hours=(72, 48, 24))
    toc = time()
    log('(MULE) TILES generation ok? {}. TOOK {:.3f} s'.format(ok, toc - tic), 'info', verbose)
    return ok, toc - tic


def _loop_rsc_generator(catalog, sleep_time, num_times, verbose=False):
    log('**RESOURCE_GENERATOR LOOP desde MULE con PID={}!!!'.format(os.getpid()), 'debug', False)
    counter = 0
    while (counter < num_times) or (num_times == 0):
        gen_ok, took = _rsc_generator(catalog, verbose=verbose)
        counter += 1
        sleep(max(sleep_time - took, 5))
        # uwsgi.async_sleep(max(sleep_time - took, 20))


def main(test_mode=False):
    """
    CLI Manager for generate svg plots as resources for the web app
    """
    p = argparse.ArgumentParser(description="EnerPI Web Resource generator")
    p.add_argument('-o', '--one', action='store_true', help='Run only one time.')
    p.add_argument('-t', '--timer', action='store', type=int, metavar='∆T', default=TILES_GENERATION_INTERVAL_LOOP,
                   help='Set periodic timer, in seconds. Default={}.'.format(TILES_GENERATION_INTERVAL_LOOP))
    p.add_argument('-n', '--num-times', action='store', type=int, metavar='∆T', default=0,
                   help='Set # of tiles generations. Default=0 (no limit).')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose mode (logging to stdout)')
    args = p.parse_args()
    # logging.debug('(MULE)-> main con args={}'.format(args))

    # Cargamos catálogo para consulta:
    cat = init_catalog(check_integrity=False, test_mode=test_mode)

    if args.one:
        _rsc_generator(cat, verbose=args.verbose)
    else:
        log('(MULE)-> Init loop with timer={} seconds'.format(args.timer), 'debug', args.verbose)
        _loop_rsc_generator(cat, args.timer, args.num_times, args.verbose)


if __name__ == '__main__':
    main()
