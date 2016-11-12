# -*- coding: utf-8 -*-
import datetime as dt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from threading import Timer
from time import sleep, time
from enerpi.base import (CONFIG, COL_TS, COLS_DATA, show_pi_temperature,
                         set_logging_conf, log, FILE_LOGGING, LOGGING_LEVEL)
from enerpi.database import init_catalog, save_raw_data, HDF_STORE
from enerpi.pisampler import random_generator, enerpi_sampler_rms, enerpi_raw_sampler, msg_to_dict, tuple_to_dict_json
from enerpi.iobroadcast import broadcast_msg, receiver_msg_generator
from enerpi.ledrgb import get_rgbled, led_info, led_alarm, blink_color


# Current meter
# TS_DATA_MS = 0  # Para maximizar el sampling (a costa de elevar demasiado la Tª de una RPI 3 -> ~80 ºC)
TS_DATA_MS = CONFIG.getint('ENERPI_SAMPLER', 'TS_DATA_MS', fallback=12)
# ∆T para el deque donde se acumulan frames
RMS_ROLL_WINDOW_SEC = CONFIG.getfloat('ENERPI_SAMPLER', 'RMS_ROLL_WINDOW_SEC', fallback=2.)
DELTA_SEC_DATA = CONFIG.getint('ENERPI_SAMPLER', 'DELTA_SEC_DATA', fallback=2)
INIT_LOG_MARK = CONFIG.get('ENERPI_SAMPLER', 'INIT_LOG_MARK', fallback='INIT')
N_COLS_SAMPLER = len(COLS_DATA + [COL_TS])  # Columnas de datos + columna de marca de tiempo

# Disk data store
N_SAMPLES_BUFFER_DISK = CONFIG.getint('ENERPI_DATA', 'TS_DATA_MS', fallback=60)
STORE_PERIODIC_CATALOG_SEC = CONFIG.getint('ENERPI_DATA', 'TS_DATA_MS', fallback=3600)

# Variable global para controlar el estado de notificaciones vía RGB LED
LED_STATE = 0

PALETA = dict(off=(0., (0., 0., 1.)),
              standby=(250., (0., 1., 0.)),
              medium=(750., (1., .5, 0.)),
              high=(3500., (1., 0., 0.)),
              max=(4500., (1., 0., 1.)))


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

    def __nonzero__(self):
        return time() - self.tic < self.timeout


def _interp_colors(c1, c2, ini_v, fin_v, value):
    color = [0] * 3
    d = value - ini_v
    assert (fin_v != ini_v)
    for i in range(3):
        p = (c2[i] - c1[i]) / (fin_v - ini_v)
        color[i] = round(c1[i] + d * p, 3)
    return tuple(color)


def _get_color(value, paleta=PALETA):
    if value >= paleta['max'][0]:
        return paleta['max'][1]
    elif value >= paleta['high'][0]:
        c_ini, c_fin = 'high', 'max'
    elif value >= paleta['medium'][0]:
        c_ini, c_fin = 'medium', 'high'
    elif value >= paleta['standby'][0]:
        c_ini, c_fin = 'standby', 'medium'
    else:
        c_ini, c_fin = 'off', 'standby'
    ini, fin = paleta[c_ini][1], paleta[c_fin][1]
    ini_v, fin_v = paleta[c_ini][0], paleta[c_fin][0]
    return _interp_colors(ini, fin, ini_v, fin_v, value)


def _reset_led_state():
    global LED_STATE
    LED_STATE = 0


def _set_led_state_alarm(led, time_blinking=2.5, timeout=3):
    global LED_STATE
    LED_STATE = 2
    if led is not None:
        led_alarm(led, time_blinking)
    if timeout > 0:
        Timer(timeout, _reset_led_state).start()


def _set_led_state_info(led, n_blinks=3):
    global LED_STATE
    LED_STATE = 1
    if led is not None:
        led_info(led, n=n_blinks)
    Timer(n_blinks + .1, _reset_led_state).start()


def _set_led_blink_rgbled(led, valor_w):
    if led is not None:
        blink_color(led, color=_get_color(valor_w), n=1)


# General
def _execfunc(func, *args_func, **kwargs_func):
    try:
        func(*args_func, **kwargs_func)
        return True
    except KeyboardInterrupt:
        # raise KeyboardInterrupt
        return False


# Receiver
def _show_cli_bargraph(d_data, ancho_disp=80, v_max=4000):
    n_resto = 3
    v_bar = min(d_data['power'], v_max) * ancho_disp / v_max
    n_big = int(v_bar)
    n_little = int(round((v_bar - int(v_bar)) / (1 / n_resto)))
    if n_little == n_resto:
        n_little = 0
        n_big += 1
    line = '⚡ {:%H:%M:%S.%f}'.format(d_data[COL_TS])[:-3] + ': '
    # if 'power' in d_data:
    line += '\033[1m{:.0f} W\033[1m; '.format(d_data['power'])
    if 'power_2' in d_data:
        line += '\033[1m{:.0f} W\033[1m; '.format(d_data['power_2'])
    if 'power_3' in d_data:
        line += '\033[1m{:.0f} W\033[1m; '.format(d_data['power_3'])
    if 'ldr' in d_data:
        line += '\033[33mLDR={:.3f} \033[32m'.format(d_data['ldr'])
    leng_intro = len(line)
    line += '◼︎' * n_big + '⇡︎' * n_little
    log(line, 'debug', True, False)
    return leng_intro


def _get_console_cols_size(len_preffix=25):
    try:
        _, columns = os.popen('stty size', 'r').read().split()
        return int(columns) - len_preffix, True
    except ValueError:
        # log('No stty! ValueError: "{}"'.format(e), 'error')
        return 120 - len_preffix, False


def _receiver(verbose=True, timeout=None):
    gen = receiver_msg_generator(verbose)
    counter_msgs, last_msg = 0, ''
    leng_intro = len('⚡ 16:10:38.326: 3433 W; LDR=0.481 ︎')
    n_cols_bar, hay_consola = _get_console_cols_size(leng_intro)
    v_max = 3500
    cond_while = True if timeout is None else TimerExiter(timeout)
    while cond_while:
        try:
            msg, delta_msg, delta_decrypt = next(gen)
            if msg != last_msg:
                counter_msgs += 1
                d_data = msg_to_dict(msg)
                try:
                    if verbose and (v_max < d_data[COLS_DATA[0]]):
                        v_max = np.ceil(d_data[COLS_DATA[0]] / 500) * 500
                        log('Se cambia la escala del CLI_bar_graph a P_MAX={:.0f} W'.format(v_max), tipo='info')
                    if verbose:
                        leng_intro = _show_cli_bargraph(d_data, ancho_disp=n_cols_bar, v_max=v_max)
                    if counter_msgs % 10 == 0:  # Actualiza tamaño de consola cada 10 samples
                        n_cols_bar, _ = _get_console_cols_size(leng_intro)
                except KeyError as e:
                    log('RECEIVER: {}. Received data: {} [MSG={}]'.format(e, d_data, msg), 'error', verbose, True)
                last_msg = msg
        except StopIteration:
            log('Terminada la recepción...', 'debug', verbose, True)
            break


def receiver(verbose=True, timeout=None):
    """
    Runs ENERPI CLI receiver

    Sample output:
    ⚡⚡ ︎ENERPI AC CURRENT SENSOR ⚡⚡
    AC Current Meter for Raspberry PI with GPIOZERO and MCP3008
    SENDER - RECEIVER vía UDP. Broadcast IP: 192.168.1.255, PORT: 57775
    01:18:02.683: 336 W; LDR=0.039 ◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎⇡︎⇡︎
    01:18:03.694: 338 W; LDR=0.039 ◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎⇡︎⇡︎
    01:18:04.704: 335 W; LDR=0.040 ◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎◼︎⇡︎⇡︎
    ...
    press CTRL+C to exit

    :param verbose:
    :param timeout:
    """
    _execfunc(_receiver, verbose=verbose, timeout=timeout)


# ENERPI logger
def _sender(data_generator, ts_data=1, path_st=HDF_STORE, timeout=None, verbose=True):

    def _save_buffer(buffer, process_save, path_store, data_catalog, v):
        if process_save is not None and process_save.is_alive():
            process_save.terminate()
        process_save = mp.Process(target=save_raw_data, args=(buffer, path_store, data_catalog, v))
        process_save.start()
        return process_save

    global LED_STATE
    LED_STATE = 0
    counter, p_save = 0, None
    normal_exit = True
    led = get_rgbled(verbose=True)
    sock_send, counter_unreachable = None, np.array([0, 0])
    catalog = init_catalog(raw_file=path_st, check_integrity=True, archive_existent=True)

    l_ini = [np.nan] * N_COLS_SAMPLER
    l_ini[0] = dt.datetime.now()
    buffer_disk = np.array(l_ini * N_SAMPLES_BUFFER_DISK).reshape(N_SAMPLES_BUFFER_DISK, N_COLS_SAMPLER)
    tic_abs = time()

    cond_while = True if timeout is None else TimerExiter(timeout)
    try:
        while cond_while:
            tic = time()
            # Recibe sample del generador de datos
            data = next(data_generator)
            if data is None:
                raise KeyboardInterrupt
            # elif verbose:
            #     print('Sampled: ', dict(zip([COL_TS] + COLS_DATA, data)), data)

            # Broadcast mensaje
            sock_send = broadcast_msg(tuple_to_dict_json(data), counter_unreachable,
                                      sock_send=sock_send, verbose=verbose)
            if (counter_unreachable[0] > 1) and (LED_STATE == 0):  # 2x blink rojo ERROR NETWORK
                _set_led_state_alarm(led, time_blinking=2.5, timeout=3)

            # Almacenamiento en buffer
            for i in range(len(data)):
                buffer_disk[counter, i] = data[i]
            counter += 1

            # Blink LED cada 2 seg
            if (LED_STATE == 0) and (counter % 2 == 0):
                _set_led_blink_rgbled(led, data[1])

            # Almacenamiento en disco del buffer
            if counter >= N_SAMPLES_BUFFER_DISK:
                # Compactado de HDF Store cada STORE_PERIODIC_CATALOG_SEC
                w_compact = time() - tic_abs >= STORE_PERIODIC_CATALOG_SEC
                if w_compact:
                    p_save = _save_buffer(buffer_disk.copy(), p_save, path_st, catalog, verbose)
                else:
                    p_save = _save_buffer(buffer_disk.copy(), p_save, path_st, None, verbose)
                if w_compact:
                    tic_abs = time()
                # 2x blink azul en grabación
                _set_led_state_info(led, n_blinks=2)
                buffer_disk[:, 1] = np.nan
                counter = 0
            toc = time()
            # Sleep cycle
            sleep(max(0.001, ts_data - (toc - tic)))
    except StopIteration:
        log('Exiting SENDER because StopIteration', 'warn', verbose)
    except KeyboardInterrupt:
        # log('Interrumpting SENDER with KeyboardInterrupt', 'warn', verbose)
        normal_exit = False
    if sock_send is not None:
        sock_send.close()
    if led is not None:
        led.close()
    if not normal_exit:
        raise KeyboardInterrupt


def _get_raw_chunk(data_generator, ts_data=1, verbose=True):
    global LED_STATE
    LED_STATE = 0
    counter, p_save = 0, None
    led = get_rgbled(verbose=True)

    tic_abs = toc = time()
    acumulador_ts = []
    acumulador_data = []
    try:
        while True:
            tic = time()
            # Recibe sample del generador de datos
            data = next(data_generator)
            if data is None:
                raise KeyboardInterrupt
            elif verbose:
                print('Sampled: ', data[0][0], data[1][0], data[0][-1], data[1][-1])

            # Acumulación en buffer
            acumulador_ts.append(data[0].copy())
            acumulador_data.append(data[1].copy())
            counter += 1

            # Blink LED cada 2 seg
            if (LED_STATE == 0) and (counter % 2 == 0):
                _set_led_blink_rgbled(led, 2000)

            toc = time()
            # Sleep cycle
            sleep(max(0.001, ts_data - (toc - tic)))
    except StopIteration:
        log('Exiting RAW SAMPLER because StopIteration', 'warn', verbose)
    except KeyboardInterrupt:
        log('Interrumpting RAW SAMPLER with KeyboardInterrupt', 'warn', verbose)
    if led is not None:
        led.close()
    # Construcción de pd.df raw_data:
    log('Making RAW data (Sampling took {:.3f} secs)'.format(toc - tic_abs), 'info', verbose)
    df = pd.DataFrame(pd.concat([pd.DataFrame(mat_v, index=pd.Series(arr_d, name='ts'),
                                              columns=['raw_0', 'raw_power', 'raw_ldr'])
                                 for arr_d, mat_v in zip(acumulador_ts, acumulador_data)])).sort_index()
    toc_abs = time()
    log('RAW data (Total time: {:.3f} secs):\n{}\n{}'.format(toc_abs - tic_abs, df.head(7), df.tail(7)), 'ok', verbose)
    return df


def _sender_random(path_st=HDF_STORE, ts_data=1, timeout=None, verbose=True):
    """
    Runs Enerpi Logger in demo mode (sends random values)

    :param path_st:
    :param ts_data:
    :param verbose:
    :return:
    """
    ok = _execfunc(_sender, random_generator(), ts_data=ts_data, path_st=path_st, timeout=timeout, verbose=verbose)
    log('Exiting SENDER_RANDOM; status:{}'.format(ok), 'info')
    return ok


def _enerpi_logger(path_st=HDF_STORE, delta_sampling=DELTA_SEC_DATA,
                   roll_time=RMS_ROLL_WINDOW_SEC, sampling_ms=TS_DATA_MS, timeout=None, verbose=True):
    """
    Runs ENERPI Sensor & Logger

    :param path_st:
    :param delta_sampling:
    :param roll_time:
    :param sampling_ms:
    :param timeout:
    :param verbose:
    :return:
    """
    s_calc = sampling_ms if sampling_ms > 0 else 8
    n_samples = int(round(roll_time * 1000 / s_calc))
    intro = (INIT_LOG_MARK + '\n  *** Calculating RMS values with window of {} frames (deltaT={} s, sampling: {} ms)'
             .format(n_samples, roll_time, sampling_ms))
    log(intro, 'ok', True)
    ok = _execfunc(_sender, enerpi_sampler_rms(n_samples_buffer=n_samples, delta_sampling=delta_sampling,
                                               min_ts_ms=sampling_ms, verbose=verbose),
                   ts_data=1, path_st=path_st, timeout=timeout, verbose=verbose)
    log('Exiting ENERPI_LOGGER; status:{}'.format(ok), 'info', verbose)
    return ok


def enerpi_raw_data(path_st, roll_time=RMS_ROLL_WINDOW_SEC, sampling_ms=TS_DATA_MS, delta_secs=20, verbose=True):
    """
    Runs ENERPI Sensor & Logger in RAW DATA mode

    :param path_st:
    :param roll_time:
    :param sampling_ms:
    :param delta_secs:
    :param verbose:
    :return:
    """
    if sampling_ms > 0:
        s_calc = sampling_ms
        n_samples = int(round(roll_time * 1000 / s_calc))
    else:
        n_samples = 5000
    intro = (INIT_LOG_MARK + '\n  *** Calculating RMS values with window of {} frames (deltaT={} s, sampling: {} ms)'
             .format(n_samples, roll_time, sampling_ms))
    intro += ('\n  ** RAW_DATA Mode ON (adquiring all samples in {} secs) (chunk={} samples) **'
              .format(delta_secs, n_samples))
    log(intro, 'ok', True)
    raw_data = _get_raw_chunk(enerpi_raw_sampler(delta_secs=delta_secs, n_samples_buffer=n_samples,
                                                 min_ts_ms=sampling_ms, verbose=verbose),
                              ts_data=0, verbose=verbose)
    if type(raw_data) is pd.DataFrame:
        log('Exiting ENERPI_RAW_LOGGER with:\n{}'.format(raw_data.describe()), 'info', verbose)
        raw_data.to_hdf(path_st, 'raw')
    return raw_data


def enerpi_logger(path_st=None, is_demo=True, timeout=None, verbose=True, **kwargs_sender):
    """
    Starts ENERPI Logger loop.
    """

    n_execs = 0
    tic = time()
    while True:
        if is_demo:
            _ = _sender_random(path_st=path_st, timeout=timeout, verbose=verbose, **kwargs_sender)
            break
        else:
            ok = _enerpi_logger(path_st=path_st, timeout=timeout, verbose=verbose, **kwargs_sender)
        if not ok:
            break
        toc = time()
        log('Waiting 10 s to init again... EXEC={}, ∆T={:.1f} s'.format(n_execs, toc - tic), 'ok', verbose, True)
        sleep(10)
        tic = time()
        n_execs += 1


def enerpi_daemon_logger(with_pitemps=False):
    """
    Punto de entrada directa a ENERPI Logger con la configuración de DATA_PATH/config_enerpi.ini.

    Se utiliza para iniciar ENERPI como daemon mediante 'enerpi-daemon start|stop|restart'
    (en conjunción con enerpiweb, con user www-data: 'sudo -u www-data %(path_env_bin)/enerpi-daemon start')

    :param with_pitemps: :bool: Logs RPI temperature every 3 seconds
    """

    set_logging_conf(FILE_LOGGING, LOGGING_LEVEL, with_initial_log=False)
    timer_temps = None
    if with_pitemps:
        # Shows RPI Temps
        timer_temps = Timer(3, show_pi_temperature, args=(3,))
        timer_temps.start()
    enerpi_logger(path_st=HDF_STORE, is_demo=False, timeout=None, verbose=False,
                  delta_sampling=DELTA_SEC_DATA, roll_time=RMS_ROLL_WINDOW_SEC, sampling_ms=TS_DATA_MS)
    if timer_temps is not None:
        log('Stopping RPI TEMPS sensing desde enerpi_main_logger...', 'debug', False, True)
        timer_temps.cancel()
