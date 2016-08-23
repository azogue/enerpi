# -*- coding: utf-8 -*-
import logging
import os
from time import time, sleep
from enerpi.database import init_catalog, DATA_PATH, HDF_STORE
from enerpi.enerplot import gen_svg_tiles
# from multiprocessing import Process
# from threading import Thread


basedir = os.path.abspath(os.path.dirname(__file__))

SERVER_FILE_LOGGING = os.path.join(basedir, 'static', 'enerpiweb_rscgen.log')
IMG_TILES_BASEPATH = os.path.join(basedir, 'static', 'img', 'generated')
TILES_GENERATION_INTERVAL = 180
LOGGING_LEVEL_SERVER = 'DEBUG'

# Establecemos logging
logging.basicConfig(filename=SERVER_FILE_LOGGING, level=LOGGING_LEVEL_SERVER, datefmt='%d/%m/%Y %H:%M:%S',
                    format='%(levelname)s [%(filename)s_%(funcName)s] - %(asctime)s: %(message)s')
logging.debug('(MULE)->LOG Estableciendo LOGFILE en {}'.format(SERVER_FILE_LOGGING))


###############################
# RESOURCE GENERATOR
###############################
def _rsc_generator(catalog):
    tic, ok = time(), False
    logging.debug('**RESOURCE_GENERATOR desde MULE con PID={}!!!'.format(os.getpid()))
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


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description="EnerPI Web Resource generator")
    p.add_argument('-o', '--one', action='store_true', help='Run only one time.')
    p.add_argument('-t', '--timer', action='store', type=int, metavar='∆T', default=TILES_GENERATION_INTERVAL,
                   help='Set periodic timer, in seconds. Default={}.'.format(TILES_GENERATION_INTERVAL))
    args = p.parse_args()
    logging.debug('(MULE)-> main con args={}'.format(args))

    # Cargamos catálogo
    cat = init_catalog(base_path=DATA_PATH, raw_file=HDF_STORE, check_integrity=False)

    if args.one:
        _rsc_generator(cat)
    else:
        logging.debug('(MULE)-> Init loop with timer={} seconds'.format(args.timer))
        print('(MULE)-> Init loop with timer={} seconds'.format(args.timer))
        _loop_rsc_generator(cat, args.timer)
    logging.debug('(MULE) Saliendo')
    # p = Process(target=_rsc_generator, args=(cat,))
    # p.start()
    # p.join()
    # _rsc_generator(cat)
