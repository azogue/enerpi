# -*- coding: utf-8 -*-
import configparser
import locale
import logging
import os
import pytz
import shutil
import subprocess
from time import time, sleep
import traceback

from enerpi import BASE_PATH, PRETTY_NAME


ENCODING = 'UTF-8'
CONFIG_FILENAME = 'config_enerpi.ini'
HAY_PP = True
try:
    # noinspection PyUnresolvedReferences
    from prettyprinting import print_err, print_red, print_info, print_ok, print_warn, print_yellowb, print_magenta
except ImportError:
    HAY_PP = False

HAY_TEMPS = True
try:
    from enerpi.pitemps import get_cpu_temp, get_gpu_temp
except ImportError:
    get_cpu_temp = get_gpu_temp = None
    logging.debug('* No se encuentra el módulo "pitemps" para medir las Tªs de la RPI')
    HAY_TEMPS = False


def funcs_tipo_output(tipo_log):
    """
    Functions for printing and logging based on type.

    :param tipo_log: :enum: error, debug, ok, info, warn, magenta
    :return: print_func, logging_func
    """
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


def log(msg, tipo, verbose=True, log_msg=True):
    """
    Logging wrapper to log and / or color-print

    :param msg:
    :param tipo:
    :param verbose: :bool:
    :param log_msg: :bool:
    """
    f1, f2 = funcs_tipo_output(tipo)
    try:
        if verbose:
            f1(msg)
        if log_msg:
            f2(msg)
    except UnicodeEncodeError as e:
        print('ERROR UnicodeEncodeError: {}'.format(e))
        traceback.print_exc(limit=5)
        print(msg.encode('UTF-8'))


def set_logging_conf(filename, level='DEBUG', verbose=True, with_initial_log=True):
    # Logging configuration
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logging.basicConfig(filename=filename, level=level, datefmt='%d/%m/%Y %H:%M:%S',
                        format='%(levelname)s [%(filename)s_%(funcName)s] - %(asctime)s: %(message)s')
    if with_initial_log:
        log(PRETTY_NAME, 'ok', verbose)


def show_pi_temperature(ts=3):
    """
    Sensor Raspberry PI temperatures, infinite-loop of logging / printing values

    :param ts:
    :return:
    """
    if HAY_TEMPS:
        while True:
            t_cpu = get_cpu_temp()
            t_gpu = get_gpu_temp()
            log('Tªs --> {:.1f} / {:.1f} ºC'.format(t_cpu, t_gpu), 'otro', False, True)
            sleep(ts)


def timeit(cadena_log, verbose=False, *args_dec):
    """
    Decorator (wrapper) to timeit and log (and print) any function.

    (For debugging / optimize purposes)

    :param cadena_log:
    :param verbose:
    :param args_dec:
    :return:
    """
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
    """
    Read (or tail) a text file.

    :param filename:
    :param tail:
    :param reverse:
    :return:
    """
    if os.path.exists(filename):
        if os.path.isfile(filename):
            try:
                if tail is not None:
                    output = subprocess.check_output(['/usr/bin/tail', '-n', '{}'.format(int(tail)), filename])
                    lines = output.decode().split('\n')
                else:
                    with open(filename, 'r', encoding=ENCODING) as file:
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


def check_resource_files(dest_path, origin_path=None):
    """
    Check needed files and directories in DATA_PATH. Init if needed (1º exec).

    :param dest_path:
    :param origin_path:
    :return bool (dest_path exists previously)
    """
    if not os.path.exists(dest_path):
        if origin_path is None:
            log('-> Making paths to "{}"'.format(dest_path), 'info', True, True)
            os.makedirs(dest_path, exist_ok=True)
        else:
            origin_path = os.path.abspath(origin_path)
            if os.path.isfile(origin_path):
                log('-> Copying resource file from "{}" to "{}"'.format(origin_path, dest_path), 'info', True, True)
                shutil.copy(origin_path, dest_path)
            else:
                log('-> Replicating tree from "{}" to "{}"'.format(origin_path, dest_path), 'info', True, True)
                shutil.copytree(origin_path, dest_path)
        log('** check_resource_files OK', 'debug', True, True)
        return False
    return True


def get_config():
    """
    Loads or generates ini file for ENERPI (& ENERPIweb) configuration.

    1) Looks for variable path DATA_PATH in file .../enerpi/config/.enerpi_data_path
    2) Tries to load 'config_enerpi.ini' from DATA_PATH, as user custom config.
    3) If not present, generates it copying the default configuration.

    :return: configparser loaded object
    """

    # Load DATA_PATH:
    dir_config = os.path.join(BASE_PATH, 'config')
    path_default_datapath = os.path.join(dir_config, '.enerpi_data_path')
    try:
        with open(path_default_datapath, 'r', encoding=ENCODING) as f:
            raw = f.read()
        data_path = raw.split('\n')[0]
        if data_path != os.path.expanduser(data_path):  # Hay '~', se expande el usuario y se graba abspath
            data_path = os.path.expanduser(data_path)
            log('''Sobreescritura del archivo "{}",
almacenando la ruta absoluta a la instalación de ENERPI
    -> DATA_PATH := {}
** Para mover la instalación de ENERPI a otro lugar, edite directamente este fichero
   (y mueva manualmente la carpeta DATA_PATH)'''.format(path_default_datapath, data_path), 'info', True, True)
            with open(path_default_datapath, 'w', encoding=ENCODING) as f:
                f.write(data_path)
    except Exception as e:
        log('ENERPI LOAD CONFIG ERROR at "{}" --> {} [{}]'
            .format(path_default_datapath, e, e.__class__), 'error', True, True)
        data_path = os.path.expanduser('~/ENERPIDATA')

    # Checks paths & re-gen if not existent
    check_resource_files(data_path)
    path_file_config = os.path.join(data_path, CONFIG_FILENAME)
    if not os.path.exists(path_file_config):
        log('** Instalando fichero de configuración en: "{}"'.format(path_file_config), 'info', True, True)
        shutil.copy(os.path.join(dir_config, 'default_config_enerpi.ini'), path_file_config)

    # Config parser
    configp = configparser.RawConfigParser()
    try:
        configp.read(path_file_config, encoding=ENCODING)
    except Exception as e:
        log('Error loading configuration INI file in "{}". Exception {} [{}]. Using defaults...'
            .format(path_file_config, e, e.__class__), 'error', True, True)
        configp.read(os.path.join(dir_config, 'default_config_enerpi.ini'), encoding=ENCODING)
    return data_path, configp


# Loads configuration
DATA_PATH, CONFIG = get_config()
TZ = pytz.timezone(CONFIG.get('ENERPI_SAMPLER', 'TZ', fallback='Europe/Madrid'))
FILE_LOGGING = os.path.join(DATA_PATH, CONFIG.get('ENERPI_DATA', 'FILE_LOGGING', fallback='enerpi.log'))
LOGGING_LEVEL = CONFIG.get('ENERPI_DATA', 'LOGGING_LEVEL', fallback='DEBUG')

# Set Locale
CUSTOM_LOCALE = CONFIG.get('ENERPI_SAMPLER', 'LOCALE', fallback='{}.{}'.format(*locale.getlocale()))
locale.setlocale(locale.LC_ALL, CUSTOM_LOCALE)
