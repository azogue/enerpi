# -*- coding: utf-8 -*-
import argparse
import logging
import os
from time import time, sleep
# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')
# noinspection PyPep8
from enerpi.base import DATA_PATH, CONFIG, check_resource_files, set_logging_conf
# noinspection PyPep8
from enerpi.database import init_catalog
# noinspection PyPep8
from enerpiplot.enerplot import gen_svg_tiles


basedir_enerweb = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'enerpiweb')
STATIC_PATH = os.path.join(DATA_PATH, CONFIG.get('ENERPI_WEBSERVER', 'STATIC_PATH'))
SERVER_FILE_LOGGING_RSCGEN = os.path.join(STATIC_PATH, 'enerpiweb_rscgen.log')
IMG_TILES_BASEPATH = os.path.join(STATIC_PATH, 'img', 'generated')
check_resource_files(STATIC_PATH, os.path.join(basedir_enerweb, 'static'))
check_resource_files(IMG_TILES_BASEPATH)
TILES_GENERATION_INTERVAL = 180
LOGGING_LEVEL_SERVER = 'DEBUG'

# Establecemos logging
set_logging_conf(SERVER_FILE_LOGGING_RSCGEN, LOGGING_LEVEL_SERVER, with_initial_log=False)
# logging.debug('(MULE_RSCGEN)->LOG Estableciendo LOGFILE en {}'.format(SERVER_FILE_LOGGING_RSCGEN))


###############################
# RESOURCE GENERATOR
###############################
def _rsc_generator(catalog):
    tic, ok = time(), False
    # logging.debug('**RESOURCE_GENERATOR desde MULE con PID={}!!!'.format(os.getpid()))
    # try:
    ok = gen_svg_tiles(IMG_TILES_BASEPATH, catalog, last_hours=(72, 48, 24))
    toc = time()
    logging.debug('(MULE) TILES generation ok? {}. TOOK {:.3f} s'.format(ok, toc - tic))
    # except Exception as e:
    #     toc = time()
    #     logging.error('Error {} [{}] en gen_svg_tiles. TOOK {:.3f} s'.format(e, e.__class__, toc - tic))
    return ok, toc - tic


def _loop_rsc_generator(catalog, sleep_time):
    logging.debug('**RESOURCE_GENERATOR LOOP desde MULE con PID={}!!!'.format(os.getpid()))
    counter = 0
    while True:
        gen_ok, took = _rsc_generator(catalog)
        counter += 1
        sleep(max(sleep_time - took, 20))
        # uwsgi.async_sleep(max(sleep_time - took, 20))


def main():
    """
    CLI Manager for generate svg plots as resources for the web app
    """
    p = argparse.ArgumentParser(description="EnerPI Web Resource generator")
    p.add_argument('-o', '--one', action='store_true', help='Run only one time.')
    p.add_argument('-t', '--timer', action='store', type=int, metavar='∆T', default=TILES_GENERATION_INTERVAL,
                   help='Set periodic timer, in seconds. Default={}.'.format(TILES_GENERATION_INTERVAL))
    args = p.parse_args()
    # logging.debug('(MULE)-> main con args={}'.format(args))

    # Cargamos catálogo para consulta:
    cat = init_catalog(check_integrity=False)

    if args.one:
        _rsc_generator(cat)
    else:
        logging.debug('(MULE)-> Init loop with timer={} seconds'.format(args.timer))
        # print('(MULE)-> Init loop with timer={} seconds'.format(args.timer))
        _loop_rsc_generator(cat, args.timer)
    # logging.debug('(MULE) Saliendo')
    # p = Process(target=_rsc_generator, args=(cat,))
    # p.start()
    # p.join()
    # _rsc_generator(cat)


if __name__ == '__main__':
    main()
