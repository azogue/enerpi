# -*- coding: utf-8 -*-
import datetime as dt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from threading import Timer
from time import sleep, time
from enerpi.base import BASE_PATH, CONFIG, DATA_PATH, log
from enerpi.database import init_catalog, save_raw_data, HDF_STORE
from enerpi.pisampler import random_generator, enerpi_sampler_rms, msg_to_dict, tuple_to_msg, COL_TS, COLS_DATA
from enerpi.iobroadcast import broadcast_msg, receiver_msg_generator
from enerpi.ledrgb import get_rgbled, led_info, led_alarm, blink_color


# Current meter
# TS_DATA_MS = 0  # Para maximizar el sampling (a costa de elevar demasiado la Tª de una RPI 3 -> ~80 ºC)
TS_DATA_MS = CONFIG.getint('ENERPI_SAMPLER', 'TS_DATA_MS', fallback=12)
# ∆T para el deque donde se acumulan frames
RMS_ROLL_WINDOW_SEC = CONFIG.getint('ENERPI_SAMPLER', 'RMS_ROLL_WINDOW_SEC', fallback=2)
DELTA_SEC_DATA = CONFIG.getint('ENERPI_SAMPLER', 'DELTA_SEC_DATA', fallback=2)
INIT_LOG_MARK = CONFIG.get('ENERPI_SAMPLER', 'INIT_LOG_MARK', fallback='INIT')
N_COLS_SAMPLER = len(COLS_DATA + [COL_TS])  # Columnas de datos + columna de marca de tiempo

# Disk data store
N_SAMPLES_BUFFER_DISK = CONFIG.getint('ENERPI_DATA', 'TS_DATA_MS', fallback=60)
STORE_PERIODIC_CATALOG_SEC = CONFIG.getint('ENERPI_DATA', 'TS_DATA_MS', fallback=3600)

# Variable global para controlar el estado de notificaciones vía RGB LED
LED_STATE = 0

# Debug variables
COLS_DEB_SEND = ['N', 'T_msg', 'T_send', 'T_buffer', 'T_disk']
COLS_DEB_RECV = ['T_msg', 'T_crypt', 'msg']
DEBUG_TIMES = []


# LED CONTROL
def _get_paleta_rgb_led():
    # color = paleta.loc[:valor_w].iloc[-1]
    return pd.read_csv(os.path.join(BASE_PATH, 'rsc', 'paleta_power_w.csv')
                       ).set_index('Unnamed: 0')['0'].str[1:-1].str.split(', ').apply(lambda x: [float(i) for i in x])


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


def _set_led_blink_paleta(led, paleta, valor_w):
    if led is not None:
        # log('blink_led POWER = {:.0f} W, color={}'.format(valor_w, PALETA_P.loc[:valor_w].iloc[-1]), 'debug', True)
        blink_color(led, color=paleta.loc[:valor_w].iloc[-1], n=1)


# General
def _execfunc(func, cols_debug, debug, *args_func, **kwargs_func):
    try:
        func(debug, *args_func, **kwargs_func)
        return True
    except KeyboardInterrupt:
        if debug and len(DEBUG_TIMES) > 0:
            if cols_debug is None:
                cols_debug = ['deb_{}'.format(i) for i in range(len(DEBUG_TIMES[0]))]

            df_tiempos = pd.DataFrame(DEBUG_TIMES, columns=cols_debug)
            path_debug = os.path.join(DATA_PATH, 'debug_tiempos__{}.csv'.format(func.__name__))
            log('FIN. TIEMPOS:\n{}\nGrabada como CSV en {}'.format(df_tiempos.describe(), path_debug), 'info', True)
            df_tiempos.to_csv(path_debug)
        # raise KeyboardInterrupt
        return False


# Receiver
def _show_cli_bargraph(d_data, log_msgs=False, ancho_disp=80, v_max=4000):
    n_resto = 3
    v_bar = min(d_data[COLS_DATA[0]], v_max) * ancho_disp / v_max
    n_big = int(v_bar)
    n_little = int(round((v_bar - int(v_bar)) / (1 / n_resto)))
    if n_little == n_resto:
        n_little = 0
        n_big += 1
    log('⚡ {:%H:%M:%S.%f}'.format(d_data[COL_TS])[:-3] + ': \033[1m{:.0f} W\033[1m; \033[33mLDR={:.3f} \033[32m'
        .format(d_data[COLS_DATA[0]], d_data[COLS_DATA[-1]]) + '◼︎' * n_big + '⇡︎' * n_little,
        'debug', True, log_msgs)


def _get_console_cols_size():
    _, columns = os.popen('stty size', 'r').read().split()
    return int(columns)


def _receiver(debug=False, verbose=True):
    gen = receiver_msg_generator(verbose)
    counter_msgs, last_msg = 0, ''
    leng_intro = len('⚡ 16:10:38.326: 3433 W; LDR=0.481 ︎')
    n_cols_bar = _get_console_cols_size() - leng_intro
    v_max = 3500
    while True:
        try:
            msg, delta_msg, delta_decrypt = next(gen)
            if msg != last_msg:
                counter_msgs += 1
                d_data = msg_to_dict(msg)
                if verbose and (v_max < d_data[COLS_DATA[0]]):
                    v_max = np.ceil(d_data[COLS_DATA[0]] / 500) * 500
                    log('Se cambia la escala del CLI_bar_graph a P_MAX={:.0f} W'.format(v_max), tipo='info')
                if verbose:
                    _show_cli_bargraph(d_data, debug, ancho_disp=n_cols_bar, v_max=v_max)
                if debug:
                    DEBUG_TIMES.append([1000 * f for f in (delta_msg, delta_decrypt)] + [d_data['msg']])
                if counter_msgs % 10 == 0:  # Actualiza tamaño de consola cada 10 samples
                    n_cols_bar = _get_console_cols_size() - leng_intro
                last_msg = msg
        except StopIteration:
            log('Terminada la recepción...', 'debug', verbose, True)
            break


def receiver(verbose=True, debug=False):
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
    :param debug:
    """
    _execfunc(_receiver, COLS_DEB_RECV, debug=debug, verbose=verbose)


# ENERPI logger
def _sender(debug, func_get_data, ts_data=1, path_st=HDF_STORE, verbose=True):

    def _save_buffer(buffer, process_save, path_store, data_catalog, v):
        if process_save is not None and process_save.is_alive():
            process_save.terminate()
        process_save = mp.Process(target=save_raw_data, args=(buffer, path_store, data_catalog, v))
        process_save.start()
        return process_save

    global LED_STATE
    LED_STATE = 0
    counter, p_save = 0, None
    led = get_rgbled(verbose=True)
    paleta_rgbled = _get_paleta_rgb_led()
    socket, counter_unreachable = None, np.array([0, 0])

    catalog = init_catalog(raw_file=path_st, check_integrity=True, archive_existent=True)

    l_ini = [np.nan] * N_COLS_SAMPLER
    l_ini[0] = dt.datetime.now()
    buffer_disk = np.array(l_ini * N_SAMPLES_BUFFER_DISK).reshape(N_SAMPLES_BUFFER_DISK, N_COLS_SAMPLER)
    tic_abs = time()

    try:
        while True:
            tic = time()
            # Recibe sample del generador de datos
            data = next(func_get_data)
            if data is None:
                raise KeyboardInterrupt
            # d_data = tuple_to_dict(data)
            toc_m = time()

            # Broadcast mensaje
            socket = broadcast_msg(tuple_to_msg(data), counter_unreachable, sock_send=socket, verbose=verbose)
            toc_n = time()
            if (counter_unreachable[0] > 1) and (LED_STATE == 0):  # 2x blink rojo ERROR NETWORK
                _set_led_state_alarm(led, time_blinking=2.5, timeout=3)

            # Almacenamiento en buffer
            for i in range(len(data)):
                buffer_disk[counter, i] = data[i]
            counter += 1
            toc_b = time()

            # Blink LED cada 2 seg
            if (LED_STATE == 0) and (counter % 2 == 0):
                _set_led_blink_paleta(led, paleta_rgbled, data[1])

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

            if debug:
                DEBUG_TIMES.append([1000 * f for f in (counter, toc_m - tic, toc_n - toc_m,
                                                       toc_b - toc_n, toc - toc_b)])
            # Sleep cycle
            sleep(max(0.001, ts_data - (toc - tic)))
    except StopIteration:
        log('SALIENDO DE SENDER', 'warn')
    except KeyboardInterrupt:
        log('Interrumpting SENDER with KeyboardInterrupt', 'warn')
        raise KeyboardInterrupt
    if socket is not None:
        socket.close()


def sender_random(ts_data=1, verbose=True, debug=True, path_st=HDF_STORE):
    """
    Runs Enerpi Logger in demo mode (sends random values)

    :param ts_data:
    :param verbose:
    :param debug:
    :param path_st:
    :return:
    """
    ok = _execfunc(_sender, COLS_DEB_SEND, debug, random_generator(), ts_data=ts_data, path_st=path_st, verbose=verbose)
    log('SALIENDO DE SENDER_RANDOM', 'info')
    return ok


def enerpi_logger(path_st=HDF_STORE,
                  delta_sampling=DELTA_SEC_DATA, roll_time=RMS_ROLL_WINDOW_SEC, sampling_ms=TS_DATA_MS,
                  verbose=True, debug=False):
    """
    Runs ENERPI Sensor & Logger

    :param path_st:
    :param delta_sampling:
    :param roll_time:
    :param sampling_ms:
    :param verbose:
    :param debug:
    :return:
    """
    s_calc = sampling_ms if sampling_ms > 0 else 8
    n_samples = int(round(roll_time * 1000 / s_calc))
    intro = (INIT_LOG_MARK + '\n  *** Haciendo RMS con window de {} frames (deltaT={} s, sampling: {} ms)'
             .format(n_samples, roll_time, sampling_ms))
    if debug:
        intro += '\n  ** DEBUG Mode ON (se grabarán tiempos) **'
    log(intro, 'ok', True)
    ok = _execfunc(_sender, COLS_DEB_SEND, debug, enerpi_sampler_rms(n_samples_buffer=n_samples,
                                                                     delta_sampling=delta_sampling,
                                                                     min_ts_ms=sampling_ms, verbose=verbose),
                   ts_data=0, path_st=path_st, verbose=verbose)
    log('SALIENDO DE ENERPI_LOGGER', 'info', verbose)
    return ok
