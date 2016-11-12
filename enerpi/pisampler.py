# -*- coding: utf-8 -*-
import datetime as dt
from gpiozero import MCP3008
import json
import numpy as np
import random
import re
from subprocess import check_output
from time import sleep, time
from enerpi.base import CONFIG, TZ, log, ANALOG_SENSORS, COL_TS, COLS_DATA, COLS_DATA_RMS, COLS_DATA_MEAN, FMT_TS

# Current meter
# Voltaje típico RMS de la instalación a medir. (SÓLO SE ESTIMA P_ACTIVA!!)
VOLTAJE = CONFIG.getint('ENERPI_SAMPLER', 'VOLTAJE', fallback=236)
# 30 A para 1 V --> Pinza amperométrica SCT030-030
A_REF = CONFIG.getfloat('ENERPI_SAMPLER', 'A_REF', fallback=30.)
# V, V_ref RPI GPIO
V_REF = CONFIG.getfloat('ENERPI_SAMPLER', 'V_REF', fallback=3.3)
# ∆T en segundos entre envíos de información (yielding)
DELTA_SEC_DATA = CONFIG.getint('ENERPI_SAMPLER', 'DELTA_SEC_DATA', fallback=1)

RMS_ROLL_WINDOW_SEC = 2  # ∆T para el deque donde se acumulan frames
N_SAMPLES_BUFFER = 250  # Nº de samples tenidos en cuenta para calcular el RMS instantáneo
PREC_SAMPLING = dt.timedelta(microseconds=500)
MEASURE_LDR_DIVISOR = CONFIG.getint('ENERPI_SAMPLER', 'MEASURE_LDR_DIVISOR', fallback=10)

HOST = check_output('hostname').decode().splitlines()[0]


class AnalogSensorBuffer(object):
    """
    Wrapper for MCP Sensor with 1D ring buffer using numpy arrays for RMS & M sensing.
    """
    def __init__(self, probe, value_correction, length, rms_sensor=True):
        self._probe = probe
        self._value_correction = value_correction
        self.size_max = length
        self._rms_sensor = rms_sensor
        self.size = 0
        self._data = np.zeros(length, dtype='f')
        self._last_out = 0.
        self._last_mean = 0.

    @property
    def software_spi(self):
        """True if MCP/GPIOZERO is using Software SPI (much slower than Hardware mode)"""
        str_repr = self._probe.__repr__()
        return 'using software SPI' in str_repr

    @property
    def is_active(self):
        """Returns GPIOZERO_MCP sensor.is_active"""
        return self._probe.is_active

    @property
    def is_rms(self):
        """True if it's a Root-Mean-Squared measuring GPIOZERO_MCP sensor"""
        return self._rms_sensor

    def append(self, x):
        """append an element to obtain future Root-Mean-Squared (RMS) value or future simple Mean value"""
        if self._rms_sensor:
            new = x**2
        else:
            new = x
        if self.size < self.size_max:
            self._last_out = self._data[self.size]
            self.size += 1
            self._last_mean += new / self.size_max
        else:
            self._last_out = self._data[-1]
            self._last_mean += (new - self._last_out) / self.size_max
        self._data = np.roll(self._data, 1)
        self._data[0] = new

    def mean(self):
        """Returns the mean of data in the ring buffer"""
        # if self.size == self.size_max:
        #     return np.mean(self._data)
        # else:
        #     return np.mean(self._data[:self.size])
        return self._last_mean

    def read(self):
        """Reads MCP sensor values, and appends value_correction"""
        return self._probe.value + self._value_correction

    def close_probe(self):
        """Closes MCP sensor"""
        if self._probe is not None and not self._probe.closed:
            self._probe.close()


def tuple_to_dict_json(data_tuple):
    """
    Makes JSON message for broadcasting, from tuple of sampling values.

    :param data_tuple: :tuple: sampling values
    :return: :str: JSON message
    """
    d_data = dict(zip([COL_TS] + COLS_DATA, data_tuple))
    d_data['host'] = HOST
    d_data[COL_TS] = d_data[COL_TS].strftime(FMT_TS)
    for c in filter(lambda x: x in d_data, COLS_DATA_RMS):
        d_data[c] = int(round(d_data[c]))
    for c in filter(lambda x: x in d_data, COLS_DATA_MEAN):
        d_data[c] = round(d_data[c], 4)
    js = json.dumps(d_data)
    return js


def msg_to_dict(msg):
    """
    Makes dict of broadcasted message of sampling values.

    :param msg: :str: message
    :return: :dict: data dict
    """
    try:
        d_data = json.loads(msg)
    except json.decoder.JSONDecodeError:
        # Versión anterior:
        rg_msg_mask = re.compile('^(?P<host>.*) __ (?P<ts>.*) __ (?P<power>.*) W __ Noise: (?P<noise>.*) W __ '
                                 'REF: (?P<ref>.*) __ LDR: (?P<ldr>.*)')
        d_data = rg_msg_mask.search(msg).groupdict()
        for k in filter(lambda x: x in d_data, COLS_DATA):
            d_data[k] = float(d_data[k])
    d_data[COL_TS] = TZ.localize(dt.datetime.strptime(d_data[COL_TS], FMT_TS))
    d_data['msg'] = msg
    return d_data


def random_generator():
    """Random data generator of sampling values for DEMO mode."""
    p_min, p_max = 180, VOLTAJE * 15
    count = 0
    while count < 50:
        p = random.randint(p_min, p_max)
        yield dt.datetime.now(), p, 1, 0, .5
        count += 1
    log('PROGRAMMED STOP OF RANDOM_GENERATOR', 'info', True, False)
    raise StopIteration


def _close_analog_sensor(sensor):
    if sensor is not None and not sensor.closed:
        sensor.close()


def _sampler(n_samples_buffer=N_SAMPLES_BUFFER, delta_sampling=DELTA_SEC_DATA, min_ts_ms=0,
             delta_secs_raw_capture=None, verbose=False):
    delta_sampling_calc = dt.timedelta(seconds=delta_sampling)
    con_pausa = min_ts_ms > 0
    buffers, normal_exit = [], True
    assert(len(ANALOG_SENSORS) > 0)
    n_samples_buffer_rms = n_samples_buffer
    n_samples_buffer_normal = n_samples_buffer // MEASURE_LDR_DIVISOR
    try:
        buffers = [AnalogSensorBuffer(MCP3008(channel=ch), bias,
                                      n_samples_buffer_rms if is_rms else n_samples_buffer_normal, rms_sensor=is_rms)
                   for ch, bias, is_rms, _ in ANALOG_SENSORS]
        software_spi_mode = buffers[0].software_spi
        log('ENERPI ANALOG SENSING WITH MCP3008 - (channel={}, raw_value={}); '
            'Active:{}, Software SPI:{}, #buffer_rms:{}, #buffer:{}'
            .format(ANALOG_SENSORS[0][0], buffers[0].read(), buffers[0].is_active, software_spi_mode,
                    n_samples_buffer_rms, n_samples_buffer_normal), 'debug', verbose)
        if software_spi_mode:
            log('SOFTWARE_SPI --> No hardware/driver present, so is going to be slower...', 'warn', verbose)
            # raise KeyboardInterrupt
        counter_frames = counter_buffer_rms = counter_buffer_normal = 0

        if delta_secs_raw_capture is not None:
            max_counter_frames = delta_secs_raw_capture * n_samples_buffer
            buffer_values = np.zeros((n_samples_buffer, len(buffers)))
            buffer_dates = np.array([np.nan] * n_samples_buffer, dtype=dt.datetime)
            tic = time()
            while counter_frames < max_counter_frames:
                buffer_values[counter_buffer_normal, :] = [b.read() for b in buffers]
                ts = dt.datetime.now()
                buffer_dates[counter_buffer_normal] = ts
                counter_buffer_normal += 1
                counter_frames += 1
                if counter_buffer_normal == n_samples_buffer:
                    yield (buffer_dates, buffer_values)
                    counter_buffer_normal = 0
                if con_pausa:
                    sleep(max(.00001, (min_ts_ms - .05) / 1000 - (time() - tic)))
                    tic = time()
        else:
            cumsum_sensors_rms = np.zeros(sum([b.is_rms for b in buffers]), dtype=float)
            cumsum_sensors_normal = np.zeros(sum([not b.is_rms for b in buffers]), dtype=float)
            assert (cumsum_sensors_rms.shape[0] + cumsum_sensors_normal.shape[0] == len(buffers))
            stop = dt.datetime.now()
            tic = time()
            while True:
                counter_buffer_rms += 1
                counter_frames += 1
                process_all_sensors = counter_buffer_rms % MEASURE_LDR_DIVISOR == 0

                # Read instant values:
                [b.append(b.read()) for b in buffers if process_all_sensors or b.is_rms]

                # Acumulation:
                cumsum_sensors_rms += [b.mean() for b in buffers if b.is_rms]
                if process_all_sensors:
                    counter_buffer_normal += 1
                    cumsum_sensors_normal += [b.mean() for b in buffers if not b.is_rms]

                # yield & reset every delta_sampling_calc:
                ts = dt.datetime.now()
                if ts - stop > delta_sampling_calc - PREC_SAMPLING:
                    stop = ts
                    power_rms_values = np.sqrt(cumsum_sensors_rms / counter_buffer_rms) * VOLTAJE * A_REF * V_REF
                    other_values = cumsum_sensors_normal / counter_buffer_normal

                    yield (ts, *power_rms_values, *other_values, counter_buffer_rms, counter_buffer_normal)
                    cumsum_sensors_rms[:] = 0
                    cumsum_sensors_normal[:] = 0
                    counter_buffer_rms = counter_buffer_normal = 0
                if con_pausa:
                    sleep(max(.00001, (min_ts_ms - .05) / 1000 - (time() - tic)))
                    tic = time()
    except KeyboardInterrupt:
        log('KeyboardInterrupt en PISAMPLER: Exiting...', 'warn', verbose)
        normal_exit = False
    except OSError as e:
        log('OSError en PISAMPLER: "{}". Terminando el generador con KeyboardInterrupt.'.format(e), 'error', verbose)
        normal_exit = False
    except (RuntimeError, AttributeError) as e:
        log('{} en PISAMPLER: "{}". Terminando el generador.'.format(e.__class__, e), 'error', verbose)
        normal_exit = False

    # Try to close sensors:
    try:
        [b.close_probe() for b in buffers]
    except Exception as e:
        log('ERROR "{}" en PISAMPLER intentando cerrar los sensores analógicos: "{}"'
            .format(e.__class__, e), 'error', verbose)
    if normal_exit:
        yield None
    else:
        raise KeyboardInterrupt


def enerpi_sampler_rms(n_samples_buffer=N_SAMPLES_BUFFER, delta_sampling=DELTA_SEC_DATA, min_ts_ms=0, verbose=False):
    """
    Generador de valores RMS de las conexiones analógicas vía MCP3008.
        - Esta función realiza el sampling de alta frecuencia y va calculando los valores RMS con un buffer (como una
        rolling mean de ventana=n_samples_buffer). Cada "delta_sampling" s, emite un mensaje con los valores calculados.
        - Mide la potencia eléctrica aplicando la conversión de voltajes correspondiente
        - Mide la intensidad luminosa de un sensor LDR (de 0. a 1.)
        - Mide el ruido de la señal (midiendo directamente un AnalogIn desconectado)
        - Mide el valor de referencia (midiendo directamente un AnalogIn puenteado a V_ref = 3.3 V)
    Devuelve, cada ∆T, una tupla con la marca temporal y las medidas en RMS

    :param n_samples_buffer: Nº de samples tenidos en cuenta para calcular el RMS instantáneo.
    :param delta_sampling: ∆T en segundos entre envíos de información (yielding)
    :param min_ts_ms: ∆T en ms mínimo entre samples. Por defecto a 0: el máximo nº de frames que pueda computarse.
    :param verbose: Salida de msgs de error por sys.stdout.

    :yield: (ts_datetime, power_rms, noise_rms, counter_buffer, ldr_rms)
    """
    return _sampler(n_samples_buffer=n_samples_buffer,
                    delta_sampling=delta_sampling, min_ts_ms=min_ts_ms, verbose=verbose)


def enerpi_raw_sampler(delta_secs=20, n_samples_buffer=N_SAMPLES_BUFFER, min_ts_ms=0,
                       verbose=True):
    """
    Generador de valores en bruto de las conexiones analógicas vía MCP3008 (sampling de alta frecuencia)

    :param delta_secs: Nº de samples tenidos en cuenta para calcular el RMS instantáneo.
    :param n_samples_buffer: Nº de samples tenidos en cuenta para hacer el yielding.
    :param min_ts_ms: ∆T en ms mínimo entre samples. Por defecto a 0: el máximo nº de frames que pueda computarse.
    :param verbose: Salida de msgs de error por sys.stdout.

    :yield: (ts_datetime, power_rms, noise_rms, counter_buffer, ldr_rms)
    """
    return _sampler(delta_secs_raw_capture=delta_secs,
                    n_samples_buffer=n_samples_buffer, min_ts_ms=min_ts_ms, verbose=verbose)
