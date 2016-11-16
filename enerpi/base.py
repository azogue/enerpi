# -*- coding: utf-8 -*-
from collections import OrderedDict
import configparser
import locale
import logging
import os
import pytz
import shutil
import subprocess
import sys
import textwrap
from threading import Timer
from time import time, sleep
from enerpi import BASE_PATH, PRETTY_NAME
from enerpi.prettyprinting import print_err, print_red, print_info, print_ok, print_warn, print_yellowb, print_magenta
from enerpi.pitemps import get_cpu_temp, get_gpu_temp


ENCODING = 'UTF-8'
CONFIG_FILENAME = 'config_enerpi.ini'


class EnerpiAnalogSensor(object):
    """
    Object for centralize access to sensor properties, as description, color, units, etc.

    """
    def __init__(self, column_name, description='', channel=0, channel_name='CH_PROBE',
                 is_rms=True, color='0CBB43', bias=0):
        self.name = column_name
        self.description = description
        self.channel = channel
        self.channel_name = channel_name
        self.is_rms = is_rms
        self.unit = 'W' if is_rms else '%'
        self.color = '#{}'.format(color)
        self.bias = bias

    def __repr__(self):
        return 'ANALOG_S: {} ({}{}, ch={}, unit={})'.format(self.name, 'RMS, ' if self.is_rms else '',
                                                            self.description, self.channel, self.unit)


class EnerpiSamplerConf(object):
    """
    Object for easy access to data properties, as # of columns, column names, descriptions and units, type of sensors,
    color for plotting, summary info & columns, etc...

    """
    def __init__(self, configparser_obj):
        # Read config INI file
        self._fmt_ts = configparser_obj.get('ENERPI_SAMPLER', 'FMT_TS', fallback='%Y-%m-%d %H:%M:%S.%f')
        self._col_ts = configparser_obj.get('ENERPI_SAMPLER', 'COL_TS', fallback='ts')
        self._ref_column_rms = 'ref'
        self._ref_column_mean = 'ref_n'

        # Voltage divisor for sct030 probes:
        mcp3008_dac_prec = 10  # bits
        niveles = 2 ** mcp3008_dac_prec - 1
        bias_current = -(niveles // 2) / niveles

        # Get columns
        cols_data_rms = configparser_obj.get('ENERPI_SAMPLER', 'COLS_DATA_RMS', fallback='power').split(', ')
        cols_data_mean = configparser_obj.get('ENERPI_SAMPLER', 'COLS_DATA_MEAN', fallback='noise, ldr').split(', ')

        # Get descriptions
        descr_data_rms = configparser_obj.get('ENERPI_SAMPLER', 'COLS_DATA_RMS', fallback='power').split(', ')
        descr_data_mean = configparser_obj.get('ENERPI_SAMPLER', 'COLS_DATA_MEAN', fallback='noise, ldr').split(', ')

        # Get colors
        colors_data_rms = configparser_obj.get('ENERPI_SAMPLER', 'COLORS_DATA_RMS', fallback='power').split(', ')
        colors_data_mean = configparser_obj.get('ENERPI_SAMPLER', 'COLORS_DATA_MEAN', fallback='noise, ldr').split(', ')
        # Get channel probes
        names_channels = list(configparser_obj['MCP3008'].keys())
        channels = [configparser_obj.getint('MCP3008', ch_name, fallback=i) for i, ch_name in enumerate(names_channels)]

        self._sensors_rms = [EnerpiAnalogSensor(name, desc, ch, ch_n, True, color, bias_current)
                             for name, desc, color, ch, ch_n in zip(cols_data_rms, descr_data_rms, colors_data_rms,
                                                                    channels[:len(cols_data_rms)],
                                                                    names_channels[:len(cols_data_rms)]) if ch >= 0]
        self._sensors_mean = [EnerpiAnalogSensor(name, desc, ch, ch_n, False, color)
                              for name, desc, color, ch, ch_n in zip(cols_data_mean, descr_data_mean, colors_data_mean,
                                                                     channels[len(cols_data_rms):],
                                                                     names_channels[len(cols_data_rms):]) if ch >= 0]
        # PiSampler parameters:
        # Current meter
        # Voltaje típico RMS de la instalación a medir. (SÓLO SE ESTIMA P_ACTIVA!!)
        voltaje = CONFIG.getint('ENERPI_SAMPLER', 'VOLTAJE', fallback=236)
        # 30 A para 1 V --> Pinza amperométrica SCT030-030
        a_ref = CONFIG.getfloat('ENERPI_SAMPLER', 'A_REF', fallback=30.)
        # V, V_ref RPI GPIO
        v_ref = CONFIG.getfloat('ENERPI_SAMPLER', 'V_REF', fallback=3.3)
        self.voltaje = voltaje
        self.rms_multiplier = self.voltaje * a_ref * v_ref

        # Sampling:
        self.ts_data_ms = CONFIG.getint('ENERPI_SAMPLER', 'TS_DATA_MS', fallback=12)
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
        log('PermissionError: {}'.format(e), 'error', True, False)
        # sudo chmod 777 ~/ENERPIDATA/enerpi.log
        # TODO Notify error
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


def show_pi_temperature(log_rpi_temps, ts=3):
    """
    Sensor Raspberry PI temperatures, infinite-loop of logging / printing values

    :param log_rpi_temps: :bool: activate RPI temp logging
    :param ts: :int: ∆T between samples
    """
    def _loop_log_temps():
        while True:
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
        return _wrapper
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


def config_dict_for_web_edit(lines_ini_file):
    """
    * Divide configuration INI file with this structure:
    [section]
    # comment
    ; comment
    VARIABLE = VALUE
    (discarding file header comments)

    Make Ordered dict like:
    ==> [(section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         (section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         ...]

    :param lines_ini_file: :list: Content of INI file after 'readlines()'
    :return: :OrderedDict:

    """
    def _is_bool_variable(value):
        value = value.lower()
        if (value == 'true') or (value == 'false'):
            return True, value == 'true'
        return False, None

    config_entries = OrderedDict()
    section, comment = None, None
    init = False
    for l in lines_ini_file:
        l = l.replace('\n', '').lstrip().rstrip()
        if l.startswith('['):
            section = l.replace('[', '').replace(']', '')
            init = True
            comment = None
            config_entries[section] = OrderedDict()
        elif l.startswith('#') or l.startswith(';'):
            if init:
                if comment is None:
                    comment = l[1:].lstrip()
                else:
                    comment += ' {}'.format(l[1:].lstrip())
        elif init and (len(l) > 0):
            # Read variable and append (/w comments)
            variable_name, variable_value = l.split('=')
            variable_name = variable_name.lstrip().rstrip()
            variable_value = variable_value.lstrip().rstrip()
            is_bool, bool_value = _is_bool_variable(variable_value)
            if not is_bool:
                try:
                    variable_value = int(variable_value)
                    var_type = 'int'
                except ValueError:
                    try:
                        variable_value = float(variable_value)
                        var_type = 'float'
                    except ValueError:
                        var_type = 'str'
            else:
                var_type = 'bool'
                variable_value = bool_value
            config_entries[section][variable_name] = variable_value, var_type, comment
            comment = None
    return config_entries


def config_changes(dict_web_form, dict_config):
    """
    Process changes in web editor

    :param dict_web_form: :OrderedDict: Posted Form values
    :param dict_config:  :OrderedDict: config dict of dicts (like the one 'config_dict_for_web_edit' returns)
    :return: :tuple: (:bool: changed, :list: updated variables, :OrderedDict: updated dict_config)

    """
    def _is_changed(value, params, name):
        if params[1] == 'int':
            try:
                value = int(value)
            except ValueError:
                value = float(value)
        elif params[1] == 'float':
            value = float(value)
        elif params[1] == 'bool':
            value = (value.lower() == 'true') or (value.lower() == 'on')
        if value != params[0]:
            log('"{}" -> HAY CAMBIO DE {} a {} (type={})'.format(name, params[0], value, params[1]), 'debug', False)
            return True, value
        return False, value

    dict_config_updated = dict_config.copy()
    dict_web_form = dict_web_form.copy()
    vars_updated = []
    for section, entries in dict_config_updated.items():
        for variable_name, variable_params in entries.items():
            if variable_name in dict_web_form:
                new_v = dict_web_form.pop(variable_name)
                changed, new_value = _is_changed(new_v, variable_params, variable_name)
                if changed:
                    vars_updated.append((variable_name, variable_params[0], new_value))
                    params_var = list(dict_config_updated[section][variable_name])
                    params_var[0] = new_value
                    dict_config_updated[section][variable_name] = tuple(params_var)
            elif (variable_params[1] == 'bool') and variable_params[0]:  # Bool en off en el form y True en config
                vars_updated.append((variable_name, variable_params[0], False))
                params_var = list(dict_config_updated[section][variable_name])
                params_var[0] = False
                log('"{}" -> HAY CHECKBOX CH DE {} a {} (type={})'
                    .format(variable_name, variable_params[0], False, variable_params[1]), 'debug', False)
                dict_config_updated[section][variable_name] = tuple(params_var)
    return len(vars_updated) > 0, vars_updated, dict_config_updated


def make_ini_file(dict_config, dest_path=None):
    """
    Makes INI file (and writes it if dest_path is not None) from an OrderedDict
    like the one 'config_dict_for_web_edit' returns.

    * INI file with this structure:
    [section]
    # comment
    ; comment
    VARIABLE = VALUE

    Ordered dict like:
    ==> [(section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         (section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         ...]

    :param dict_config: :OrderedDict: Content of INI file after 'readlines()'
    :param dest_path: :str: optional path for INI file write.
    :return: :str: raw INI text

    """
    lines = ['# -*- coding: utf-8 -*-']
    for section, entries in dict_config.items():
        lines.append('[{}]'.format(section.upper()))
        for variable_name, (value, var_type, comment) in entries.items():
            if comment:
                [lines.append('# {}'.format(l_wrap)) for l_wrap in textwrap.wrap(comment, 80)]
            lines.append('{} = {}'.format(variable_name, value))
        lines.append('')  # Separador entre secciones
    ini_text = '\n'.join(lines)
    if dest_path is not None:
        _makedirs(dest_path)
        with open(dest_path, 'w') as f:
            f.write(ini_text)
    return ini_text


def check_resource_files(dest_path, origin_path=None):
    """
    Check needed files and directories in DATA_PATH. Init if needed (1º exec).

    :param dest_path:
    :param origin_path:
    :return bool (dest_path exists previously)
    """
    if not os.path.exists(dest_path):
        _makedirs(dest_path)
        if origin_path is None:
            log('-> Made paths to "{}"'.format(dest_path), 'info', True, True)
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
    elif oct(os.stat(dest_path).st_mode)[-3:] != '777':
        os.chmod(dest_path, 0o777)
    return True


def _get_config():
    """
    Loads or generates ini file for ENERPI (& ENERPIweb) configuration.

    1) Looks for variable path DATA_PATH in file .../enerpi/config/.enerpi_data_path
    2) Tries to load 'config_enerpi.ini' from DATA_PATH, as user custom config.
    3) If not present, generates it copying the default configuration.

    :return: :configparser: loaded object
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
        check_resource_files(path_file_config)
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
DATA_PATH, CONFIG = _get_config()
FILE_LOGGING = os.path.join(DATA_PATH, CONFIG.get('ENERPI_DATA', 'FILE_LOGGING', fallback='enerpi.log'))
LOGGING_LEVEL = CONFIG.get('ENERPI_DATA', 'LOGGING_LEVEL', fallback='DEBUG')

# Set Locale
CUSTOM_LOCALE = CONFIG.get('ENERPI_SAMPLER', 'LOCALE', fallback='{}.{}'.format(*locale.getlocale()))
locale.setlocale(locale.LC_ALL, CUSTOM_LOCALE)

# ANALOG SENSORS WITH MCP3008 (Rasp.io Analog Zero)
SENSORS = EnerpiSamplerConf(CONFIG)
