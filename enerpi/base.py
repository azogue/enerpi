# -*- coding: utf-8 -*-
"""
ENERPI - Common base objects:
- Base classes, methods and decorators
- Configuration loaders
- Constant definitions
...

"""
import configparser
import json
import locale
import logging
import os
import pytz
import shutil
import subprocess
import sys
from threading import Thread, Timer
from time import time, sleep
from enerpi import BASE_PATH, PRETTY_NAME
from enerpi.prettyprinting import print_err, print_red, print_info, print_ok, print_warn, print_yellowb, print_magenta
from enerpi.pitemps import get_cpu_temp, get_gpu_temp


ENCODING = 'UTF-8'
CONFIG_FILENAME = 'config_enerpi.ini'
SENSORS_CONFIG_JSON_FILENAME = 'sensors_enerpi.json'


class TimerExiter(object):
    """
    Simple timer, without threading (and without any precission!),
    for limiting time in infinite loops (receiver & emitter)

    It has overloaded the __nonzero__ method, for asking it in while loops like:
        t = TimerExiter(7)
        while t:
            do cool things...
    """

    def __init__(self, timeout):
        self.timeout = timeout
        self.tic = time()

    def __bool__(self):
        return time() - self.tic < self.timeout

    __nonzero__ = __bool__


class EnerpiAnalogSensor(object):
    """
    Object for centralize access to sensor properties, as description, color, units, etc.

    """
    def __init__(self, column_name, description='', channel=0, is_rms=True, unit='W', color='#0CBB43', bias=0,
                 tile_gradient_st='', tile_gradient_end='', icon='bolt'):
        self.name = column_name
        self.description = description
        self.channel = channel
        self.is_rms = is_rms
        self.unit = unit
        self.color = color
        self.bias = bias
        self.icon = icon
        self.tile_gradient_st = tile_gradient_st
        self.tile_gradient_end = tile_gradient_end

    def __repr__(self):
        str_type = 'RMS, ' if self.is_rms else ''
        return ('ANALOG_S: {} ({}{}, ch={}, unit={}, color={}, bias={:.3f})'
                .format(self.name, str_type, self.description, self.channel, self.unit, self.color, self.bias))

    def to_dict(self):
        """
        Return dict with analog sensor properties for jinja2 templating

        :return: dict

        """
        return dict(name=self.name, description=self.description, channel=self.channel, is_rms=self.is_rms,
                    unit=self.unit, color=self.color, bias=self.bias,
                    icon=self.icon, tile_gradient_st=self.tile_gradient_st, tile_gradient_end=self.tile_gradient_end)


def _load_analog_sensors(sensors_json_conf):
    """
    Load list of 'EnerpiAnalogSensor' objects, with the ENERPI sensors user configuration
    Format of json file:
    [ {
        "name": "power",
        "analog_channel": 3,
        "is_rms": true,
        "unit": "W",
        "color": "#0CBB43",
        "tile_gradient_st": [12, 187, 67, 0.83],
        "tile_gradient_end": [55, 245, 119, 0.27],
        "icon": "flash",
        "description": "Main Power"
      }, {...}, ... ]

    :param sensors_json_conf: list of dicts (from JSON config file)
    :return list of analog sensors

    """
    def _get_rms_sensor_bias(sensor):
        if sensor['is_rms']:
            if 'bias' in sensor:
                return sensor['bias']
            # Voltage divisor for sct030 probes:
            mcp3008_dac_prec = 10  # bits
            niveles = 2 ** mcp3008_dac_prec - 1
            bias_current = -(niveles // 2) / niveles
            return bias_current
        return 0.

    analog_s = [EnerpiAnalogSensor(s['name'], s['description'], s['analog_channel'], s['is_rms'], s['unit'], s['color'],
                                   _get_rms_sensor_bias(s), s['tile_gradient_st'], s['tile_gradient_end'], s['icon'])
                for s in sensors_json_conf]  # JSON file sensor order
    return analog_s


class EnerpiSamplerConf(object):
    """
    Object for easy access to data properties, as # of columns, column names, descriptions and units, type of sensors,
    color for plotting, summary info & columns, etc...

    """
    def __init__(self, configparser_obj, sensors_json_conf):
        # Read config INI file
        self._fmt_ts = configparser_obj.get('ENERPI_SAMPLER', 'FMT_TS', fallback='%Y-%m-%d %H:%M:%S.%f')
        self._col_ts = configparser_obj.get('ENERPI_SAMPLER', 'COL_TS', fallback='ts')
        self._ref_column_rms = 'ref'
        self._ref_column_mean = 'ref_n'

        # Get sensors:
        sensors = _load_analog_sensors(sensors_json_conf)
        self._sensors_rms = [s for s in sensors if s.is_rms]
        self._sensors_mean = [s for s in sensors if not s.is_rms]

        # PiSampler parameters (Current meter):
        # Voltaje típico RMS de la instalación a medir. (SÓLO SE ESTIMA P_ACTIVA!!)
        voltaje = CONFIG.getint('ENERPI_SAMPLER', 'VOLTAJE', fallback=236)
        # 30 A para 1 V --> Pinza amperométrica SCT030-030
        a_ref = CONFIG.getfloat('ENERPI_SAMPLER', 'A_REF', fallback=30.)
        # V, V_ref RPI GPIO
        v_ref = CONFIG.getfloat('ENERPI_SAMPLER', 'V_REF', fallback=3.3)
        self.voltaje = voltaje
        self.rms_multiplier = self.voltaje * a_ref * v_ref

        # Sampling:
        self.ts_data_ms = CONFIG.getfloat('ENERPI_SAMPLER', 'TS_DATA_MS', fallback=12)
        # ∆T para el deque donde se acumulan frames
        self.rms_roll_window_sec = CONFIG.getfloat('ENERPI_SAMPLER', 'RMS_ROLL_WINDOW_SEC', fallback=2.)
        s_calc = self.ts_data_ms if self.ts_data_ms > 0 else 8
        self.n_samples_buffer_rms = int(round(self.rms_roll_window_sec * 1000 / s_calc))

        self.delta_sec_data = CONFIG.getfloat('ENERPI_SAMPLER', 'DELTA_SEC_DATA', fallback=1.)
        self.measure_ldr_divisor = CONFIG.getint('ENERPI_SAMPLER', 'MEASURE_LDR_DIVISOR', fallback=10)
        self.n_samples_buffer_mean = self.n_samples_buffer_rms // self.measure_ldr_divisor
        self.TZ = pytz.timezone(CONFIG.get('ENERPI_SAMPLER', 'TZ', fallback='Europe/Madrid'))

    def __repr__(self):
        repr_str = 'ENERPI SAMPLER SENSORS:\n * '
        repr_str += '\n * '.join([str(s) for s in self])
        repr_str += ('\n --SAMPLING: V={} V; ∆T={} s; sampling={} ms, N={}, {}(/{})'
                     .format(self.voltaje, self.delta_sec_data, self.ts_data_ms,
                             self.n_samples_buffer_rms, self.n_samples_buffer_mean, self.measure_ldr_divisor))
        return repr_str

    def __len__(self):
        return len(self._sensors_rms) + len(self._sensors_mean)

    def __getitem__(self, key):
        if type(key) is int:
            if key >= len(self._sensors_rms):
                try:
                    return self._sensors_mean[key - len(self._sensors_rms)]
                except IndexError:
                    raise KeyError('Sensor #{} not present (available sensors: {})'
                                   .format(key, self.columns_sensors[1:]))
            return self._sensors_rms[key]
        else:
            assert type(key) is str
            try:
                int_key = self.columns_sensors.index(key) - 1
                return self.__getitem__(int_key)
            except ValueError:
                raise KeyError("Sensor '{}' not present (available sensors: {})".format(key, self.columns_sensors[1:]))

    def __iter__(self):
        for x in self._sensors_rms + self._sensors_mean:
            yield x

    def to_dict(self):
        """
        Return dict of analog sensors for jinja2 templating

        :return: dict

        """
        d_sensors = {'sensors': [x.to_dict() for x in self._sensors_rms + self._sensors_mean]}
        return d_sensors

    @property
    def main_column(self):
        """
        First (& principal) RMS sensor (Main power)

        :return: :str: name of main column
        """
        return self._sensors_rms[0].name

    @property
    def ts_column(self):
        """
        Name of time-series index

        :return: :str: name of time-series index column
        """
        return self._col_ts

    @property
    def ts_fmt(self):
        """
        String formatting of timestamps:

        :return: :str: fmt
        """
        return self._fmt_ts

    @property
    def ref_rms(self):
        """
        Name of column with # of samples of RMS values

        :return: :str: column name
        """
        return self._ref_column_rms

    @property
    def ref_mean(self):
        """
        Name of column with # of samples of MEAN values

        :return: :str: column name
        """
        return self._ref_column_mean

    @property
    def n_cols_sensors(self):
        """
        # of columns: index column (TS) + RMS sensors + MEAN sensors

        :return: :int: # of columns
        """
        return len(self) + 1

    @property
    def n_cols_sampling(self):
        """
        # of columns: index column (TS) + RMS sensors + MEAN sensors + ref_RMS + ref_MEAN

        :return: :int: # of columns
        """
        return len(self) + 3

    @property
    def columns_sensors(self):
        """
        List of columns names of sensors (+ts): index column (TS) + RMS sensors + MEAN sensors

        :return: :list:
        """
        return [self.ts_column] + [s.name for s in self._sensors_rms + self._sensors_mean]

    @property
    def columns_sensors_rms(self):
        """
        List of columns names of RMS sensors

        :return: :list:
        """
        return [s.name for s in self._sensors_rms]

    @property
    def columns_sensors_mean(self):
        """
        List of columns names of MEAN sensors

        :return: :list:
        """
        return [s.name for s in self._sensors_mean]

    @property
    def columns_sampling(self):
        """
        List of columns names of sampling data index column (TS) + RMS sensors + MEAN sensors + REF_RMS + REF_MEAN

        :return: :list:
        """
        return self.columns_sensors + [self.ref_rms, self.ref_mean]

    def included_columns_sensors(self, dict_sample):
        """
        Return RMS & MEAN columns included in dict sample

        :return: :tuple: tuple with 2 lists (rms & mean included columns)

        """
        cols_rms = [s.name for s in self._sensors_rms if s.name in dict_sample]
        cols_mean = [s.name for s in self._sensors_mean if s.name in dict_sample]
        return cols_rms, cols_mean

    def included_columns_sampling(self, dict_sample):
        """
        Return RMS & MEAN columns (or ref_RMS and ref_MEAN) included in sampling data dict

        :return: :tuple: tuple with 3 lists (rms, mean & ref included columns)

        """
        cols_rms, cols_mean = self.included_columns_sensors(dict_sample)
        cols_ref = [c for c in [self._ref_column_rms, self._ref_column_mean] if c in dict_sample]
        return cols_rms, cols_mean, cols_ref

    def descriptions(self, columns, return_list=True):
        """
        Return column descriptions (as list or as dict c:desc)
        :param columns: list of columns
        :param return_list: :bool: False for dict return
        :return: :list: or :dict:
        """
        def _get_desc(key):
            try:
                return self[key].description
            except KeyError as e:
                if key == self.ts_column:
                    return 'TS'
                elif key == self.ref_rms:
                    return '# samples'
                elif key == self.ref_mean:
                    return '# samples_n'
                else:
                    raise KeyError(e)

        if return_list:
            return [_get_desc(c) for c in columns]
        else:
            return {c: _get_desc(c) for c in columns}


def _makedirs(dest_path):
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if os.path.exists(dest_path):
            os.utime(dest_path, None)
    except PermissionError as e:
        msg_error = '_makedirs PermissionError (dest_path = {}). Exception: {}. Exit in 2 secs...'.format(dest_path, e)
        log(msg_error, 'error', True, False)
        sys.exit(2)
    except OSError as e:
        msg_error = '_makedirs OSError (dest_path = {}). Exception: {}. Exit in 2 secs...'.format(dest_path, e)
        log(msg_error, 'error', True, False)
        sys.exit(2)


def _funcs_tipo_output(tipo_log):
    """
    Functions for printing and logging based on type.

    :param tipo_log: :enum: error, debug, ok, info, warn, magenta
    :return: print_func, logging_func
    """
    if tipo_log == 'error':
        return print_err, logging.error
    elif tipo_log == 'debug':
        return print_red, logging.debug
    elif tipo_log == 'ok':
        return print_ok, logging.info
    elif tipo_log == 'info':
        return print_info, logging.info
    elif (tipo_log == 'warn') or (tipo_log == 'warning'):
        return print_warn, logging.warning
    elif tipo_log == 'magenta':
        return print_magenta, logging.warning
    else:
        return print_yellowb, logging.debug


def log(msg, tipo, verbose=True, log_msg=True):
    """
    Logging wrapper to log and / or color-print

    :param msg:
    :param tipo:
    :param verbose: :bool:
    :param log_msg: :bool:
    """
    f1, f2 = _funcs_tipo_output(tipo)
    if verbose:
        f1(msg)
    if log_msg:
        f2(msg)


def set_logging_conf(filename, level='DEBUG', verbose=True, with_initial_log=True):
    """Logging configuration"""
    _makedirs(filename)
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    # try:
    logging.basicConfig(filename=filename, level=level, datefmt='%d/%m/%Y %H:%M:%S',
                        format='%(levelname)s [%(filename)s_%(funcName)s] - %(asctime)s: %(message)s')
    # except PermissionError as e:
    #     log('PermissionError: {}'.format(e), 'error', True, False)
    #     sudo chmod 777 ~/ENERPIDATA/enerpi.log
    #     sys.exit(2)
    if with_initial_log:
        log(PRETTY_NAME, 'ok', verbose)


def show_pi_temperature(log_rpi_temps, ts=3, timeout=None):
    """
    Sensor Raspberry PI temperatures, infinite-loop (or timeout) of logging / printing values

    :param bool log_rpi_temps: activate RPI temp logging
    :param int ts: ∆T between samples
    :param int timeout: :int: ∆T between samples

    """
    def _loop_log_temps():
        cond_while = True if timeout is None else TimerExiter(timeout)
        while cond_while:
            t_cpu = get_cpu_temp()
            t_gpu = get_gpu_temp()
            if (t_cpu is None) and (t_gpu is None):
                log('NO RPI TEMPS', 'warning', False, True)
                break
            log('Tªs --> {:.1f} / {:.1f} ºC'.format(t_cpu, t_gpu), 'otro', False, True)
            sleep(ts)

    if log_rpi_temps:
        # Shows RPI Temps
        timer_temps = Timer(.5, _loop_log_temps)
        timer_temps.start()
        return timer_temps
    return None


def async_task(f):
    """
    Decorator (wrapper) for execute an async action (threaded)
    :param f:

    """
    def _wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
    return _wrapper


def timeit(cadena_log, verbose=False, *args_dec):
    """
    Decorator (wrapper) to timeit and log (and print) any function.

    (For debugging / optimize purposes)

    :param cadena_log:
    :param verbose:
    :param args_dec:
    :return:
    """
    def _real_deco(function):
        def _wrapper(*args, **kwargs):
            kwargs_print = {}
            tic = time()
            out = function(*args, **kwargs)
            if verbose:
                print_yellowb(cadena_log.format(*args_dec, **kwargs_print) + ' TOOK: {:.3f} s'.format(time() - tic))
            logging.debug(cadena_log.format(*args_dec, **kwargs_print) + ' TOOK: {:.3f} s'.format(time() - tic))
            return out
        f = _wrapper
        f.__name__ = function.__name__
        f.__doc__ = function.__doc__
        # f.__dict__ = function.__dict__
        # f.__module__ = function.__module__
        # f.__annotations__ = function.__annotations__
        # f.__defaults__ = function.__defaults__
        # f.__kwdefaults__ = function.__kwdefaults__
        return f
    return _real_deco


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


def check_resource_files(dest_path, origin_path=None, verbose=True):
    """
    Check needed files and directories in DATA_PATH. Init if needed (1º exec).

    :param dest_path:
    :param origin_path:
    :param verbose:

    """
    if not os.path.exists(dest_path):
        _makedirs(dest_path)
        if origin_path is not None:
            origin_path = os.path.abspath(origin_path)
            if os.path.isfile(origin_path):
                log('-> Copying resource file from "{}" to "{}"'.format(origin_path, dest_path), 'info', verbose)
                shutil.copy(origin_path, dest_path)
            else:
                log('-> Replicating tree from "{}" to "{}"'.format(origin_path, dest_path), 'info', verbose)
                shutil.copytree(origin_path, dest_path)
        log('** check_resource_files OK', 'debug', verbose)
    elif oct(os.stat(dest_path).st_mode)[-3:] != '777':
        log('Permission trouble in "{}" --> {}'
            .format(dest_path, oct(os.stat(dest_path).st_mode)[-3:]), 'warning', verbose)
        try:
            os.chmod(dest_path, 0o777)
        except PermissionError:
            log('Permission error in "{}" --> {}. Cant set 777'
                .format(dest_path, oct(os.stat(dest_path).st_mode)[-3:]), 'error', verbose)


def get_config_enerpi():
    """
    Loads or generates ini file for ENERPI (& ENERPIweb) configuration.

    1) Looks for variable path DATA_PATH in file .../enerpi/config/.enerpi_data_path
    2) Tries to load 'config_enerpi.ini' from DATA_PATH, as user custom config.
    2) Tries to load 'sensors_enerpi.json' from DATA_PATH, as user custom settings for analog sensors.
    3) If not present, generates it copying the default configuration.

    :return: :str: data_path, :configparser: loaded object, :dict:, enerpi sensors configuration
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

    # Checks paths & re-gen configuration if not existent
    check_resource_files(data_path)

    # Config parser
    path_file_config = os.path.join(data_path, CONFIG_FILENAME)
    if not os.path.exists(path_file_config):
        check_resource_files(path_file_config)
        log('** Instalando fichero de configuración en: "{}"'.format(path_file_config), 'info', True, True)
        shutil.copy(os.path.join(dir_config, 'default_config_enerpi.ini'), path_file_config)
    configp = configparser.RawConfigParser(allow_no_value=True)
    try:
        configp.read(path_file_config, encoding=ENCODING)
    except Exception as e:
        log('Error loading configuration INI file in "{}". Exception {} [{}]. Using defaults...'
            .format(path_file_config, e, e.__class__), 'error', True, True)
        configp.read(os.path.join(dir_config, 'default_config_enerpi.ini'), encoding=ENCODING)

    # Sensors
    path_file_sensors_config = os.path.join(data_path, SENSORS_CONFIG_JSON_FILENAME)
    if not os.path.exists(path_file_sensors_config):
        check_resource_files(path_file_sensors_config)
        log('** Instalando fichero de configuración en: "{}"'.format(path_file_sensors_config), 'info', True, True)
        shutil.copy(os.path.join(dir_config, 'default_sensors_enerpi.json'), path_file_sensors_config)
    try:
        config_s = json.load(open(path_file_sensors_config, encoding=ENCODING))
    except Exception as e:
        log('Error loading sensors JSON configuration file in "{}". Exception {} [{}]. Using defaults...'
            .format(path_file_sensors_config, e, e.__class__), 'error', True, True)
        config_s = json.load(open(os.path.join(dir_config, 'default_sensors_enerpi.json'), encoding=ENCODING))
    return data_path, configp, config_s


def reload_config():
    """
    Method used in tests and in situations when ENERPI configuration changes.
    Read & load global config variables: SENSORS, DATA_PATH, CONFIG

    """
    global SENSORS, DATA_PATH, CONFIG

    DATA_PATH, CONFIG, new_sensors_theme = get_config_enerpi()
    SENSORS = EnerpiSamplerConf(CONFIG, new_sensors_theme)


# Loads configuration
DATA_PATH, CONFIG, sensors_theme = get_config_enerpi()

# Appends ENERPI gmail account: (Hardcoded)
GMAIL_ACCOUNT = 'enerpi.bot@gmail.com'
GMAIL_APP_PASSWORD = 'qkdspbhmxouzrkvv'

# Admin email for reports & nofifications:
RECIPIENT = CONFIG.get('NOTIFY', 'RECIPIENT', fallback='eugenio.panadero@gmail.com')

# Set Locale
custom_locale = CONFIG.get('ENERPI_SAMPLER', 'LOCALE', fallback='{}.{}'.format(*locale.getlocale()))
locale.setlocale(locale.LC_ALL, custom_locale)

# ANALOG SENSORS WITH MCP3008 (Rasp.io Analog Zero)
SENSORS = EnerpiSamplerConf(CONFIG, sensors_theme)

# Logging files & other common paths:
FILE_LOGGING = os.path.join(DATA_PATH, CONFIG.get('ENERPI_DATA', 'FILE_LOGGING', fallback='enerpi.log'))
LOGGING_LEVEL = CONFIG.get('ENERPI_DATA', 'LOGGING_LEVEL', fallback='DEBUG')

STATIC_PATH = os.path.join(DATA_PATH, CONFIG.get('ENERPI_WEBSERVER', 'STATIC_PATH', fallback='WWW'))
SERVER_FILE_LOGGING_RSCGEN = os.path.join(STATIC_PATH, 'enerpiweb_rscgen.log')
NGINX_CONFIG_FILE = 'enerpiweb_nginx.conf'
UWSGI_CONFIG_FILE = 'enerpiweb_uwsgi.ini'

IMG_TILES_BASEPATH = os.path.join(STATIC_PATH, 'img', 'generated')
IMG_BASEPATH = os.path.join(DATA_PATH, CONFIG.get('ENERPI_DATA', 'IMG_BASEPATH'))
DEFAULT_IMG_MASK = CONFIG.get('ENERPI_DATA', 'DEFAULT_IMG_MASK', fallback='enerpi_plot_{:%Y%m%d_%H%M}.png')
COLOR_TILES = (1, 1, 1)

DAEMON_STDOUT = '/tmp/enerpi_out.txt'
DAEMON_STDERR = '/tmp/enerpi_err.txt'
DAEMON_PIDFILE = '/tmp/enerpilogger.pid'
INDEX_DATA_CATALOG = 'data_catalog.csv'
