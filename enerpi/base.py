# -*- coding: utf-8 -*-
import logging
from time import time, sleep


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
            log('Tªs --> {:.1f} / {:.1f} ºC'.format(t_cpu, t_gpu), 'otro', True, True)
            sleep(ts)


def timeit(cadena_log, *args_dec):
    def real_deco(function):
        def wrapper(*args, **kwargs):
            kwargs_print = {}
            for k in kwargs.keys():
                if k.startswith('print_'):
                    kwargs_print[k] = kwargs.pop(k)
            tic = time()
            out = function(*args, **kwargs)
            print_yellowb(cadena_log.format(*args_dec, **kwargs_print) + ' TOOK: {:.3f} s'.format(time() - tic))
            return out
        return wrapper
    return real_deco


