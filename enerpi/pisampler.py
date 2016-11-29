# -*- coding: utf-8 -*-
import datetime as dt
from gpiozero import MCP3008
import json
import numpy as np
import random
import re
from subprocess import check_output
from time import sleep, time
from enerpi.base import SENSORS, log


PREC_SAMPLING = .0002  # := dt.timedelta(microseconds=500)
MIN_NUM_SAMPLES_RMS = 10
HOST = check_output('hostname').decode().splitlines()[0]
EXAMPLE_POWER_EV = [(0, 300), (3, 1200), (6, 3250), (8, 700), (9, 100), (10, 0), (12, 300)]


class DummySensor(object):
    """
    Object for mimic a MCP Sensor.
    """
    def __init__(self, channel=0, power_noise=15,
                 power_divisor=SENSORS.rms_multiplier, power_evolution=EXAMPLE_POWER_EV,
                 signal_center=.49951124, freq_hz=50):
        self._ch = channel
        self._v_min = 0
        self._v_max = 1
        self._counter = 0
        self._center = signal_center
        self._power_divisor = power_divisor
        self._noise = power_noise / self._power_divisor
        times, values = list(zip(*power_evolution))
        self._true_value = values[-1] / self._power_divisor
        self._power_ev_time = np.array(times)
        self._power_ev_values = np.array(values)
        self._period_ev = self._power_ev_time[-1] + 1
        self._freq = freq_hz
        self._t0 = time()

    @property
    def elapsed(self):
        """Elapsed time from creation"""
        return time() - self._t0

    @property
    def is_active(self):
        """True, like GPIOZERO_MCP sensor.is_active"""
        return True

    @property
    def value(self):
        """Mimics reading of MCP sensor"""
        ts = self.elapsed % self._period_ev
        valid_values = np.where(self._power_ev_time < ts)[0]
        if len(valid_values) > 0:
            value = self._power_ev_values[valid_values[-1]]
        else:
            value = -1
        self._true_value = value / self._power_divisor
        noise_v = 2 * (random.random() - .5) * self._noise
        v = self._true_value + noise_v
        return self._center + np.sqrt(2) * v * np.sin((self._freq * 2 * np.pi) * self.elapsed)

    @property
    def closed(self):
        """Mimics closed state of MCP sensor"""
        return True


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
        self._completed = False
        self._data = np.zeros(length, dtype='f')
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

    def append_reading(self):
        """Read MCP sensor values, and append with value_correction
        to obtain future Root-Mean-Squared (RMS) value or future simple Mean value"""
        x = self._probe.value + self._value_correction
        if self._rms_sensor:
            new = x**2
        else:
            new = x
        if not self._completed:
            self.size += 1
            self._last_mean += new / self.size_max
            if self.size == self.size_max:
                self._completed = True
        else:
            self._last_mean += (new - self._data[-1]) / self.size_max
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
    d_data = dict(zip(SENSORS.columns_sampling, data_tuple))
    d_data['host'] = HOST
    d_data[SENSORS.ts_column] = d_data[SENSORS.ts_column].strftime(SENSORS.ts_fmt)
    # d_data[SENSORS.ts_column] = dt.datetime.now().strftime(SENSORS.ts_fmt)

    cols_rms, cols_mean, cols_ref = SENSORS.included_columns_sampling(d_data)
    for c in cols_rms:
        d_data[c] = int(round(d_data[c]))
    for c in cols_mean:
        d_data[c] = round(d_data[c], 4)
    for c in cols_ref:
        d_data[c] = int(d_data[c])
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
        for k in filter(lambda x: x in d_data, SENSORS.columns_sensors_rms + SENSORS.columns_sensors_mean):
            d_data[k] = float(d_data[k])
    d_data[SENSORS.ts_column] = SENSORS.TZ.localize(dt.datetime.strptime(d_data[SENSORS.ts_column], SENSORS.ts_fmt))
    d_data['msg'] = msg
    return d_data


def _sampler(n_samples_buffer=SENSORS.n_samples_buffer_rms, delta_sampling=SENSORS.delta_sec_data,
             min_ts_ms=SENSORS.ts_data_ms, delta_secs_raw_capture=None,
             measure_ldr_divisor=SENSORS.measure_ldr_divisor,
             use_dummy_sensors=False, reset_bias=False, verbose=False):
    delta_sampling_calc = delta_sampling
    con_pausa = min_ts_ms > 0
    buffers = []
    assert(len(SENSORS) > 0)
    if reset_bias:
        for s in SENSORS:
            s.bias = 0
    n_samples_buffer_rms = n_samples_buffer
    n_samples_buffer_normal = n_samples_buffer // measure_ldr_divisor
    if use_dummy_sensors:
        sensor_probe, probe_type = DummySensor, 'DummySensor'
    else:
        sensor_probe, probe_type = MCP3008, 'MCP3008'

    try:
        buffers = [AnalogSensorBuffer(sensor_probe(channel=s.channel), s.bias,
                                      n_samples_buffer_rms if s.is_rms else n_samples_buffer_normal,
                                      rms_sensor=s.is_rms) for s in SENSORS]

        software_spi_mode = (probe_type is 'DummySensor') or buffers[0].software_spi
        log('ENERPI ANALOG SENSING WITH {} - (channel={}, raw_value={:.5f}); '
            'Active:{}, Software SPI:{}, #buffer_rms:{}, #buffer:{}'
            .format(probe_type, SENSORS[0].channel, buffers[0].read(), buffers[0].is_active, software_spi_mode,
                    n_samples_buffer_rms, n_samples_buffer_normal), 'debug', verbose)
        if software_spi_mode and (probe_type is 'MCP3008'):
            log('SOFTWARE_SPI --> No hardware/driver present, so bye bye...', 'error', verbose)
            raise KeyboardInterrupt
        counter_buffer_rms = counter_buffer_normal = 0

        if delta_secs_raw_capture is not None:
            max_counter_frames = delta_secs_raw_capture * n_samples_buffer
            counter_frames = 0
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
            cumsum_sensors_normal = other_values = np.zeros(sum([not b.is_rms for b in buffers]), dtype=float)
            assert (cumsum_sensors_rms.shape[0] + cumsum_sensors_normal.shape[0] == len(buffers))
            stop = tic = time()
            while True:
                counter_buffer_rms += 1
                process_all_sensors = counter_buffer_rms % measure_ldr_divisor == 0
                if process_all_sensors:
                    counter_buffer_normal += 1
                    # Read instant values:
                    [b.append_reading() for b in buffers]
                    cumsum_sensors_rms += [b.mean() for b in buffers if b.is_rms]
                    cumsum_sensors_normal += [b.mean() for b in buffers if not b.is_rms]
                else:
                    # Read instant values:
                    [b.append_reading() for b in buffers if b.is_rms]
                    cumsum_sensors_rms += [b.mean() for b in buffers if b.is_rms]

                # yield & reset every delta_sampling_calc:
                ts = time()
                if (ts - stop > delta_sampling_calc - PREC_SAMPLING) and counter_buffer_rms > MIN_NUM_SAMPLES_RMS:
                    stop = ts
                    power_rms_values = np.sqrt(cumsum_sensors_rms / counter_buffer_rms) * SENSORS.rms_multiplier
                    if counter_buffer_normal > 0:
                        other_values = cumsum_sensors_normal / counter_buffer_normal
                    yield (dt.datetime.now(), *power_rms_values, *other_values,
                           counter_buffer_rms, counter_buffer_normal)
                    cumsum_sensors_rms[:] = 0
                    cumsum_sensors_normal[:] = 0
                    counter_buffer_rms = counter_buffer_normal = 0
                elif con_pausa:
                    t_sleep = (min_ts_ms - .05) / 1000 - (ts - tic)
                    if t_sleep > .00005:
                        sleep(t_sleep)
                    tic = time()
    except KeyboardInterrupt:
        log('KeyboardInterrupt en PISAMPLER: Exiting at {}...'.format(dt.datetime.now()), 'warn', verbose)
    except OSError as e:
        log('OSError en PISAMPLER: "{}". Terminando el generador con KeyboardInterrupt.'.format(e), 'error', verbose)
    except (RuntimeError, AttributeError) as e:
        log('{} en PISAMPLER: "{}". Terminando el generador.'.format(e.__class__, e), 'error', verbose)
    # Exiting
    [b.close_probe() for b in buffers]
    raise KeyboardInterrupt


def enerpi_sampler_rms(n_samples_buffer=SENSORS.n_samples_buffer_rms,
                       delta_sampling=SENSORS.delta_sec_data, min_ts_ms=SENSORS.ts_data_ms,
                       use_dummy_sensors=False, verbose=False):
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
    :param use_dummy_sensors: Utiliza un generador de datos por software en lugar de usar MCP3008 (para DEMO / tests).
    :param verbose: Salida de msgs de error por sys.stdout.

    :yield: (ts_datetime, power_rms, noise_rms, counter_buffer, ldr_rms)
    """
    return _sampler(n_samples_buffer=n_samples_buffer,
                    delta_sampling=delta_sampling, min_ts_ms=min_ts_ms,
                    use_dummy_sensors=use_dummy_sensors, verbose=verbose)


def enerpi_raw_sampler(delta_secs=20, n_samples_buffer=SENSORS.n_samples_buffer_rms, min_ts_ms=0,
                       use_dummy_sensors=False, verbose=True):
    """
    Generador de valores en bruto de las conexiones analógicas vía MCP3008 (sampling de alta frecuencia)

    :param delta_secs: Nº de samples tenidos en cuenta para calcular el RMS instantáneo.
    :param n_samples_buffer: Nº de samples tenidos en cuenta para hacer el yielding.
    :param min_ts_ms: ∆T en ms mínimo entre samples. Por defecto a 0: el máximo nº de frames que pueda computarse.
    :param use_dummy_sensors: Utiliza un generador de datos por software en lugar de usar MCP3008 (para DEMO / tests).
    :param verbose: Salida de msgs de error por sys.stdout.

    :yield: (ts_datetime, power_rms, noise_rms, counter_buffer, ldr_rms)
    """
    return _sampler(delta_secs_raw_capture=delta_secs,
                    n_samples_buffer=n_samples_buffer, min_ts_ms=min_ts_ms,
                    use_dummy_sensors=use_dummy_sensors, reset_bias=True, verbose=verbose)
