# -*- coding: utf-8 -*-
from collections import deque
import datetime as dt
from gpiozero import MCP3008
from math import sqrt
import numpy as np
import random
import re
from subprocess import check_output
from time import sleep, time
from enerpi.base import CONFIG, TZ, log


# Conexiones analógicas vía MCP3008
MCP3008_DAC_PREC = 10  # bits
CH_VREF = CONFIG.getint('MCP3008', 'CH_VREF', fallback=0)
CH_PROBE = CONFIG.getint('MCP3008', 'CH_PROBE', fallback=4)
CH_NOISE = CONFIG.getint('MCP3008', 'CH_NOISE', fallback=3)
CH_LDR = CONFIG.getint('MCP3008', 'CH_LDR', fallback=7)
USE_NUMPY_BUFFER = CONFIG.getboolean('MCP3008', 'USE_NUMPY_BUFFER', fallback=False)
MEASURE_LDR_WITH_RMS = CONFIG.getboolean('MCP3008', 'MEASURE_LDR_WITH_RMS', fallback=False)
MEASURE_LDR_DIVISOR = CONFIG.getint('MCP3008', 'MEASURE_LDR_DIVISOR', fallback=10)

# Current meter
# Voltaje típico RMS de la instalación a medir. (SÓLO SE ESTIMA P_ACTIVA!!)
VOLTAJE = CONFIG.getint('ENERPI_SAMPLER', 'VOLTAJE', fallback=236)
# 30 A para 1 V --> Pinza amperométrica SCT030-030
A_REF = CONFIG.getfloat('ENERPI_SAMPLER', 'A_REF', fallback=30.)
# V, V_ref RPI GPIO
V_REF = CONFIG.getfloat('ENERPI_SAMPLER', 'V_REF', fallback=3.3)
# ∆T en segundos entre envíos de información (yielding)
DELTA_SEC_DATA = CONFIG.getint('ENERPI_SAMPLER', 'DELTA_SEC_DATA', fallback=1)

# RMS_ROLL_WINDOW_SEC = 2  # ∆T para el deque donde se acumulan frames
N_SAMPLES_BUFFER = 250  # Nº de samples tenidos en cuenta para calcular el RMS instantáneo
PREC_SAMPLING = dt.timedelta(microseconds=500)

# Sampling a texto y viceversa
HOST = check_output('hostname').decode().splitlines()[0]
MSG_MASK = '{} __ {:%Y-%m-%d %H:%M:%S.%f} __ {:.0f} W __ Noise: {:.6f} W __ REF: {:.3f} __ LDR: {:.4f}'
RG_MSG_MASK = re.compile('^(?P<host>.*) __ (?P<ts>.*) __ (?P<power>.*) W __ Noise: (?P<noise>.*) W __ '
                         'REF: (?P<ref>.*) __ LDR: (?P<ldr>.*)')

# Nombres de columna en pd.DataFrames y formato de fecha
COL_TS = CONFIG.get('ENERPI_SAMPLER', 'COL_TS', fallback='ts')
FMT_TS = CONFIG.get('ENERPI_SAMPLER', 'FMT_TS', fallback='%Y-%m-%d %H:%M:%S.%f')
COLS_DATA = CONFIG.get('ENERPI_SAMPLER', 'COLS_DATA', fallback='power, noise, ref, ldr').split(', ')


class NumpyBuffer(object):
    """1D ring buffer using numpy arrays for RMS & M sensing"""

    def __init__(self, length, rms_sensor=True, dtype='f'):
        self.size_max = length
        self.rms_sensor = rms_sensor
        self.size = 0
        self._data = np.zeros(length, dtype=dtype)
        self._last_out = 0.
        self._last_mean = 0.

    def append(self, x):
        """append an element to obtain future Root-Mean-Squared (RMS) value or future simple Mean value"""
        if self.rms_sensor:
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


def tuple_to_msg(data_tuple):
    return MSG_MASK.format(HOST, *data_tuple)


# def tuple_to_dict(data_tuple):
#     return dict(zip([COL_TS] + COLS_DATA, data_tuple))


def msg_to_dict(msg, func_err=print, *args_err):
    try:
        d_data = RG_MSG_MASK.search(msg).groupdict()
        d_data[COL_TS] = TZ.localize(dt.datetime.strptime(d_data[COL_TS], FMT_TS))
        for k in COLS_DATA:
            d_data[k] = float(d_data[k])
        d_data['msg'] = msg
    except AttributeError:
        d_data = {'msg': msg}
        func_err('RECEIVED ERROR mask -->"{}"'.format(RG_MSG_MASK, d_data), *args_err)
        # _log('RECEIVED ERROR mask -->"{}"'.format(RG_MSG_MASK, d_data), 'error', True)
    return d_data


def random_generator():
    p_min, p_max = 180, VOLTAJE * 15
    count = 0
    while count < 500:
        p = random.randint(p_min, p_max)
        yield dt.datetime.now(), p, 1, 0, .5
        count += 1
    log('PARADA PROGRAMADA DE RANDOM_GENERATOR', 'info', True, False)


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
    delta_sampling_calc = dt.timedelta(seconds=delta_sampling)
    con_pausa = min_ts_ms > 0
    niveles = 2 ** MCP3008_DAC_PREC - 1
    resta_bias = (niveles // 2) / niveles
    reading_probe = reading_ldr = reading_noise = None
    try:
        reading_probe = MCP3008(channel=CH_PROBE)
        # reading_ref = MCP3008(channel=CH_VREF)
        reading_ldr = MCP3008(channel=CH_LDR)
        reading_noise = MCP3008(channel=CH_NOISE)

        if MEASURE_LDR_WITH_RMS:
            n_samples_buffer_ldr = n_samples_buffer
        else:
            n_samples_buffer_ldr = n_samples_buffer // MEASURE_LDR_DIVISOR
        if USE_NUMPY_BUFFER:
            buffer = NumpyBuffer(n_samples_buffer, dtype='f')
            # buffer_ref = NumpyBuffer(n_samples_buffer, dtype='f')
            buffer_ldr = NumpyBuffer(n_samples_buffer_ldr, rms_sensor=MEASURE_LDR_WITH_RMS, dtype='f')
            buffer_noise = NumpyBuffer(n_samples_buffer, dtype='f')
        else:
            buffer = deque(np.zeros(n_samples_buffer), n_samples_buffer)
            # buffer_ref = deque(np.zeros(n_samples_buffer), n_samples_buffer)
            buffer_ldr = deque(np.zeros(n_samples_buffer_ldr), n_samples_buffer_ldr)
            buffer_noise = deque(np.zeros(n_samples_buffer), n_samples_buffer)

        # counter_frames = counter_buffer = cumsum = cumsum_ref = cumsum_ldr = cumsum_noise = 0
        counter_frames = counter_buffer = cumsum = cumsum_ldr = cumsum_noise = 0
        stop = dt.datetime.now()
        tic = time()
        while True:
            counter_buffer += 1
            counter_frames += 1
            # v = (reading_probe.value - resta_bias) / reading_ref.value
            v = reading_probe.value
            v -= resta_bias
            # v_ref = reading_ref.value
            if MEASURE_LDR_WITH_RMS or counter_buffer % MEASURE_LDR_DIVISOR == 0:
                v_ldr = reading_ldr.value
            v_noise = reading_noise.value
            ts = dt.datetime.now()
            if USE_NUMPY_BUFFER:
                buffer.append(v)
                cumsum += buffer.mean()
                buffer_noise.append(v_noise)
                cumsum_noise += buffer_noise.mean()
                # buffer_ref.append(v_ref)
                # cumsum_ref += buffer_ref.mean()
                if MEASURE_LDR_WITH_RMS or counter_buffer % MEASURE_LDR_DIVISOR == 0:
                    buffer_ldr.append(v_ldr)
                    cumsum_ldr += buffer_ldr.mean()
            else:
                buffer.append(v ** 2)
                buffer_noise.append(v_noise ** 2)
                # buffer_ref.append(v_ref ** 2)
                if MEASURE_LDR_WITH_RMS or counter_buffer % MEASURE_LDR_DIVISOR == 0:
                    buffer_ldr.append(v_ldr ** 2)

                if counter_frames < n_samples_buffer:
                    cumsum += np.mean([buffer[i] for i in range(-counter_frames, 0)])
                    # cumsum_ref += np.mean([buffer_ref[i] for i in range(-counter_frames, 0)])
                    if MEASURE_LDR_WITH_RMS or counter_buffer % MEASURE_LDR_DIVISOR == 0:
                        cumsum_ldr += np.mean([buffer_ldr[i] for i in range(-(counter_frames // MEASURE_LDR_DIVISOR), 0)])
                    cumsum_noise += np.mean([buffer_noise[i] for i in range(-counter_frames, 0)])
                else:
                    cumsum += np.mean(buffer)
                    # cumsum_ref += np.mean(buffer_ref)
                    if MEASURE_LDR_WITH_RMS or counter_buffer % MEASURE_LDR_DIVISOR == 0:
                        cumsum_ldr += np.mean(buffer_ldr)
                    cumsum_noise += np.mean(buffer_noise)
            if ts - stop > delta_sampling_calc - PREC_SAMPLING:
                stop = ts
                power_rms = sqrt(cumsum / counter_buffer) * VOLTAJE * A_REF * V_REF
                noise_rms = sqrt(cumsum_noise / counter_buffer) * VOLTAJE * A_REF * V_REF
                if MEASURE_LDR_WITH_RMS:
                    ldr = sqrt(cumsum_ldr / counter_buffer)
                else:
                    ldr = cumsum_ldr / (counter_buffer // MEASURE_LDR_DIVISOR)
                yield (ts, power_rms, noise_rms, counter_buffer, ldr)
                counter_buffer = cumsum = cumsum_ldr = cumsum_noise = 0
                # counter_buffer = cumsum = cumsum_ref = cumsum_ldr = cumsum_noise = 0
            if con_pausa:
                sleep(max(.00001, (min_ts_ms - .05) / 1000 - (time() - tic)))
                tic = time()
            # if counter_frames > 100:
            #     raise RuntimeError
    except OSError as e:
        log('OSError en PISAMPLER: "{}". Terminando el generador con KeyboardInterrupt.'.format(e), 'error', verbose)
        try:
            if reading_probe is not None:
                reading_probe.close()
            # reading_ref.close()
            if reading_ldr is not None:
                reading_ldr.close()
            if reading_noise is not None:
                reading_noise.close()
        except Exception as e:
            log('ERROR "{}" en PISAMPLER intentando cerrar los sensores analógicos: "{}"'
                .format(e.__class__, e), 'error', verbose)
        raise KeyboardInterrupt
    except (RuntimeError, AttributeError) as e:
        log('{} en PISAMPLER: "{}". Terminando el generador.'.format(e.__class__, e), 'error', verbose)
    yield None
