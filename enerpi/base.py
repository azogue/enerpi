# -*- coding: utf-8 -*-
import configparser
import logging
import os
import pytz
import subprocess
from time import time, sleep
import sys
from enerpi import BASE_PATH


CONFIG = configparser.RawConfigParser()
CONFIG.read(os.path.join(BASE_PATH, 'config_enerpi.ini'))
TZ = pytz.timezone(CONFIG.get('ENERPI_SAMPLER', 'TZ', fallback='Europe/Madrid'))
DATA_PATH = os.path.expanduser(CONFIG.get('ENERPI_DATA', 'DATA_PATH_OSX' if sys.platform == 'darwin' else 'DATA_PATH'))


HAY_PP = True
try:
    # noinspection PyUnresolvedReferences
    from prettyprinting import print_err, print_red, print_info, print_ok, print_warn, print_yellowb, print_magenta
except ImportError:
    HAY_PP = False

HAY_TEMPS = True
try:
    from pitemps import get_cpu_temp, get_gpu_temp
except ImportError:
    get_cpu_temp = get_gpu_temp = None
    print('* No se encuentra el módulo "pitemps" para medir las Tªs de la RPI')
    HAY_TEMPS = False


def funcs_tipo_output(tipo_log):
    if HAY_PP:
        if tipo_log == 'error':
            return print_err, logging.error
        elif tipo_log == 'debug':
            return print_red, logging.debug
        elif tipo_log == 'ok':
            return print_ok, logging.info
        elif tipo_log == 'info':
            return print_info, logging.info
        elif tipo_log == 'warn':
            return print_warn, logging.warning
        elif tipo_log == 'magenta':
            return print_magenta, logging.warning
        else:
            return print_yellowb, logging.debug
    else:
        if tipo_log == 'error':
            return print, logging.error
        elif tipo_log == 'debug':
            return print, logging.debug
        elif tipo_log == 'ok':
            return print, logging.info
        elif tipo_log == 'info':
            return print, logging.info
        elif tipo_log == 'warn':
            return print, logging.warning
        else:
            return print, logging.debug


def log(msg, tipo='error', verbose=True, log_msg=True):
    f1, f2 = funcs_tipo_output(tipo)
    if verbose:
        f1(msg)
    if log_msg:
        f2(msg)


def show_pi_temperature(ts=3):
    if HAY_TEMPS:
        while True:
            t_cpu = get_cpu_temp()
            t_gpu = get_gpu_temp()
            log('Tªs --> {:.1f} / {:.1f} ºC'.format(t_cpu, t_gpu), 'otro', False, True)
            sleep(ts)


def timeit(cadena_log, verbose=False, *args_dec):
    def real_deco(function):
        def wrapper(*args, **kwargs):
            kwargs_print = {}
            for k in kwargs.keys():
                if k.startswith('print_'):
                    kwargs_print[k] = kwargs.pop(k)
            tic = time()
            out = function(*args, **kwargs)
            if verbose:
                print_yellowb(cadena_log.format(*args_dec, **kwargs_print) + ' TOOK: {:.3f} s'.format(time() - tic))
            logging.debug(cadena_log.format(*args_dec, **kwargs_print) + ' TOOK: {:.3f} s'.format(time() - tic))
            return out
        return wrapper
    return real_deco


def get_lines_file(filename, tail=None, reverse=False):
    if os.path.exists(filename):
        if os.path.isfile(filename):
            try:
                if tail is not None:
                    output = subprocess.check_output(['/usr/bin/tail', '-n', '{}'.format(int(tail)), filename])
                    lines = output.decode().split('\n')
                else:
                    with open(filename, 'r') as file:
                        lines = file.read().split('\n')
                if len(lines) > 0 and lines[-1] == '':
                    lines = lines[:-1]
                if reverse:
                    return list(reversed(lines))
                return lines
            except Exception as e:
                return ['ERROR Reading {}: "{}" [{}]'.format(filename, e, e.__class__)]
        else:
            return ['{} is not a file!!'.format(filename)]
    return ['Path not found: "{}"'.format(filename)]
