# -*- coding: utf-8 -*-
import datetime as dt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from threading import Timer
from time import sleep, time
from enerpi.base import (CONFIG, SENSORS, DATA_PATH, TimerExiter, show_pi_temperature,
                         set_logging_conf, log, FILE_LOGGING, LOGGING_LEVEL)
from enerpi.database import init_catalog, save_raw_data, HDF_STORE_PATH
from enerpi.pisampler import enerpi_sampler_rms, enerpi_raw_sampler, msg_to_dict, tuple_to_dict_json
from enerpi.iobroadcast import get_codec, broadcast_msg, receiver_msg_generator
from enerpi.ledrgb import get_rgbled, led_info, led_alarm, blink_color
from enerpi.notifier import push_enerpi_error


# Disk data store
N_SAMPLES_BUFFER_DISK = CONFIG.getint('ENERPI_DATA', 'N_SAMPLES_BUFFER_DISK', fallback=60)
STORE_PERIODIC_CATALOG_SEC = CONFIG.getint('ENERPI_DATA', 'STORE_PERIODIC_CATALOG_SEC', fallback=3600)
INIT_LOG_MARK = CONFIG.get('ENERPI_SAMPLER', 'INIT_LOG_MARK', fallback='INIT')

# Variable global para controlar el estado de notificaciones vía RGB LED
LED_STATE = 0

PALETA = dict(off=(0., (0., 0., 1.)),
              standby=(250., (0., 1., 0.)),
              medium=(750., (1., .5, 0.)),
              high=(3500., (1., 0., 0.)),
              max=(4500., (1., 0., 1.)))

# Decay control (# of samples/s dropping a lot sometimes--> y = -a * x**2. x(9:32)=1300, x(9:37)=1190, x(9:48)=995)
MIN_N_SAMPLES_DELTA_FRACTION = .6
MIN_N_SAMPLES_MAX_CONSIDERED = 900
MIN_N_SAMPLES_ABS = 60
STATIC_PATH = os.path.join(DATA_PATH, CONFIG.get('ENERPI_WEBSERVER', 'STATIC_PATH'))
IMG_TILES_BASEPATH = os.path.join(STATIC_PATH, 'img', 'generated')


def _get_color(value, paleta=PALETA):
    # Interpolate color with power value

    def _interp_colors(c1, c2, ini_v, fin_v, v):
        color = [0] * 3
        d = v - ini_v
        assert (fin_v != ini_v)
        for i in range(3):
            p = (c2[i] - c1[i]) / (fin_v - ini_v)
            color[i] = round(c1[i] + d * p, 3)
        return tuple(color)

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
    ini_val, fin_val = paleta[c_ini][0], paleta[c_fin][0]
    return _interp_colors(ini, fin, ini_val, fin_val, value)


def _reset_led_state():
    global LED_STATE
    LED_STATE = 0


def _set_led_state_alarm(led, time_blinking=2.5, timeout=3, time_on=.25, alarm_type='error'):
    global LED_STATE
    LED_STATE = 2
    if led is not None:
        color = (1, 0, 0) if alarm_type == 'error' else (0, 1, 1)
        led_alarm(led, time_blinking, timeout, time_on=time_on, color=color)
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


def _show_cli_bargraph(d_data, ancho_disp=80, v_max=4000):
    # CLI TEXT Receiver bargraph
    n_resto = 3
    v_bar = min(d_data[SENSORS.main_column], v_max) * ancho_disp / v_max
    n_big = int(v_bar)
    n_little = int(round((v_bar - int(v_bar)) / (1 / n_resto)))
    if n_little == n_resto:
        n_little = 0
        n_big += 1
    line = '⚡ {:%H:%M:%S.%f}'.format(d_data[SENSORS.ts_column])[:-3] + ': '
    cols_rms, cols_mean, cols_ref = SENSORS.included_columns_sampling(d_data)
    line += '\033[1m({}) \033[1m'.format(d_data[SENSORS.ref_rms])
    for c in cols_rms:
        line += '\033[1m{:.0f} W\033[1m; '.format(d_data[c])

    line += '\033[33mm({}) \033[32m'.format(d_data[SENSORS.ref_mean])
    for c in cols_mean:
        line += '\033[33m{}={:.3f} \033[32m'.format(c, d_data[c])
    leng_intro = len(line)
    line += '◼︎' * n_big + '⇡︎' * n_little
    log(line, 'debug', True, False)
    return leng_intro


def receiver(verbose=True, timeout=None, port=None):
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

    :param verbose: :bool: Prints broadcast IP & PORT.
    :param timeout: :int: (Optional) # of seconds receiving msgs
    :param port: :int: (Optional) set broadcast PORT

    """
    def _get_console_cols_size(len_preffix=25):
        try:
            _, columns = os.popen('stty size', 'r').read().split()
            return int(columns) - len_preffix, True
        except ValueError:
            # log('No stty! ValueError: "{}"'.format(e), 'error')
            return 120 - len_preffix, False

    gen = receiver_msg_generator(verbose=verbose, port=port, codec=get_codec())
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
                    if verbose and (v_max < d_data[SENSORS.main_column]):
                        v_max = np.ceil(d_data[SENSORS.main_column] / 500) * 500
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


def enerpi_raw_data(path_st=None, roll_time=SENSORS.rms_roll_window_sec, sampling_ms=SENSORS.ts_data_ms, delta_secs=20,
                    use_dummy_sensors=False, key_save='rms', verbose=True):
    """
    Runs ENERPI Sensor & Logger in RAW DATA mode

    :param path_st:
    :param roll_time:
    :param sampling_ms:
    :param delta_secs:
    :param use_dummy_sensors:
    :param key_save:
    :param verbose:
    :return: raw_data pandas DataFrame

    """

    def _get_raw_chunk(data_generator, ts_data=1):
        # global LED_STATE
        # LED_STATE = 0
        counter, p_save = 0, None
        # led = get_rgbled(verbose=True)

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
                # if (LED_STATE == 0) and (counter % 2 == 0):
                #     _set_led_blink_rgbled(led, 2000)

                toc = time()
                # Sleep cycle
                sleep(max(0.001, ts_data - (toc - tic)))
        except StopIteration:
            log('Exiting RAW SAMPLER because StopIteration', 'warn', verbose)
        except KeyboardInterrupt:
            log('Interrumpting RAW SAMPLER with KeyboardInterrupt', 'warn', verbose)
        # if led is not None:
        #     led.close()
        # Construcción de pd.df raw_data:
        log('Making RAW data (Sampling took {:.3f} secs)'.format(toc - tic_abs), 'info', verbose)
        df = pd.DataFrame(pd.concat([pd.DataFrame(mat_v, index=pd.Series(arr_d, name='ts'),
                                                  columns=SENSORS.columns_sensors[1:])
                                     for arr_d, mat_v in zip(acumulador_ts, acumulador_data)])).sort_index()
        toc_abs = time()
        log('RAW data (Total time: {:.3f} secs):\n{}\n{}'.format(toc_abs - tic_abs, df.head(7), df.tail(7)), 'ok',
            verbose)
        return df

    if sampling_ms > 0:
        s_calc = sampling_ms
        n_samples = int(round(roll_time * 1000 / s_calc))
    else:
        n_samples = 5000
    intro = (INIT_LOG_MARK + '\n  *** Calculating RMS values with window of {} frames (deltaT={} s, sampling: {} ms)'
             .format(n_samples, roll_time, sampling_ms))
    intro += ('\n  ** RAW_DATA Mode ON (adquiring all samples in {} secs) (chunk={} samples) **'
              .format(delta_secs, n_samples))
    log(intro, 'ok', verbose)

    generator = enerpi_raw_sampler(delta_secs=delta_secs, n_samples_buffer=n_samples, min_ts_ms=sampling_ms,
                                   use_dummy_sensors=use_dummy_sensors, verbose=verbose)
    raw_data = _get_raw_chunk(generator, ts_data=0)
    if (type(raw_data) is pd.DataFrame) and (path_st is not None):
        log('Exiting ENERPI_RAW_LOGGER with:\n{}\n** SAVING DATA with key="{}" in "{}"'
            .format(raw_data.describe(), key_save, path_st), 'info', verbose)
        raw_data.to_hdf(path_st, key_save)
    return raw_data


def enerpi_logger(path_st=HDF_STORE_PATH, delta_sampling=SENSORS.delta_sec_data,
                  roll_time=SENSORS.rms_roll_window_sec, sampling_ms=SENSORS.ts_data_ms,
                  timeout=None, is_demo=False, verbose=True):
    """
    Runs ENERPI Sensor & Logger

    :param path_st:
    :param delta_sampling:
    :param roll_time:
    :param sampling_ms:
    :param timeout:
    :param is_demo:
    :param verbose:

    """
    def _save_buffer(buffer, process_save, path_store, data_catalog, v):
        if process_save is not None and process_save.is_alive():
            process_save.terminate()
        process_save = mp.Process(target=save_raw_data, args=(buffer, path_store, data_catalog, v),
                                  name='enerpi_save_buffer')
        process_save.start()
        return process_save

    global LED_STATE
    s_calc = sampling_ms if sampling_ms > 0 else 1
    n_samples_buffer = int(round(roll_time * 1000 / s_calc))
    intro = (INIT_LOG_MARK + '\n  *** Calculating RMS values with window of {} frames (deltaT={} s, sampling: {} ms)'
             .format(n_samples_buffer, roll_time, sampling_ms))
    log(intro, 'ok')

    data_generator = enerpi_sampler_rms(n_samples_buffer=n_samples_buffer, delta_sampling=delta_sampling,
                                        min_ts_ms=sampling_ms, use_dummy_sensors=is_demo, verbose=verbose)
    LED_STATE = 0
    counter, p_save = 0, None
    led = get_rgbled(verbose=verbose)
    sock_send, counter_unreachable = None, np.array([0, 0])

    catalog = init_catalog(sensors=SENSORS, raw_file=path_st, check_integrity=True, archive_existent=True)

    l_ini = [np.nan] * SENSORS.n_cols_sampling
    l_ini[0] = dt.datetime.now()
    buffer_disk = np.array(l_ini * N_SAMPLES_BUFFER_DISK).reshape(N_SAMPLES_BUFFER_DISK, SENSORS.n_cols_sampling)
    tic_abs = time()

    cond_while = True if timeout is None else TimerExiter(timeout)
    codec = get_codec()
    port = CONFIG.getint('BROADCAST', 'UDP_PORT', fallback=57775)

    if n_samples_buffer is None:
        min_n_raw_samples = MIN_N_SAMPLES_ABS
    else:
        min_n_raw_samples = max(MIN_N_SAMPLES_ABS,
                                int(MIN_N_SAMPLES_DELTA_FRACTION * min(MIN_N_SAMPLES_MAX_CONSIDERED, n_samples_buffer)))

    error_decay = {'counter_act': 0,
                   'subject': 'SAMPLING DECAY -> {}',
                   'mask': 'Sampling freq decay until {}. # act: {}. # Unreach. Net: {}',
                   'last_error_decay': None}
    # Loop
    try:
        while cond_while:
            # Recibe sample del generador de datos
            data = next(data_generator)

            # Broadcast mensaje
            sock_send = broadcast_msg(tuple_to_dict_json(data), counter_unreachable,
                                      sock_send=sock_send, verbose=verbose, codec=codec, port=port)
            # Almacenamiento en buffer
            for i in range(len(data)):
                buffer_disk[counter, i] = data[i]
            counter += 1

            if (data[-2] < MIN_N_SAMPLES_ABS) or (data[-2] < min_n_raw_samples) and (LED_STATE == 0):
                # Sampling decay problem --> red blink (ERROR SAMPLING)
                _set_led_state_alarm(led, time_blinking=10, timeout=10, time_on=.1, alarm_type='error')
                error_decay['counter_act'] += 1
                if ((error_decay['counter_act'] > 5) and
                        ((error_decay['last_error_decay'] is None)
                         or (time() - error_decay['last_error_decay'] > 300))):
                    # Decay error update
                    error_decay['last_error_decay'] = time()
                    msg = error_decay['mask'].format(data[-2], error_decay['counter_act'], counter_unreachable)
                    _ = push_enerpi_error(error_decay['subject'].format(data[-2]), msg)
                    error_decay['counter_act'] = 0
            elif (counter_unreachable[0] > 1) and (LED_STATE == 0):
                # Unreachable network (wifi issues) -->   2x yellow blink (ERROR NETWORK)
                _set_led_state_alarm(led, time_blinking=2, time_on=.2, timeout=2, alarm_type='warning')
            elif (LED_STATE == 0) and (counter % 2 == 0):
                # Normal operation: blink with color as function of main_rms value in W --> LED Blink every 2 sec
                _set_led_blink_rgbled(led, data[1])
            elif LED_STATE == 0:
                # Reset decay error with normal operation
                error_decay['counter_act'] = 0

            # Almacenamiento en disco del buffer
            if counter >= N_SAMPLES_BUFFER_DISK:
                # Compactado de HDF Store cada STORE_PERIODIC_CATALOG_SEC
                w_compact = time() - tic_abs >= STORE_PERIODIC_CATALOG_SEC
                if w_compact:
                    p_save = _save_buffer(buffer_disk, p_save, path_st, catalog, verbose)
                else:
                    p_save = _save_buffer(buffer_disk, p_save, path_st, None, verbose)
                if w_compact:
                    tic_abs = time()
                # 2x blink azul en grabación
                _set_led_state_info(led, n_blinks=2)
                buffer_disk[:, 1] = np.nan
                counter = 0
    except StopIteration:
        log('Exiting SENDER because StopIteration', 'warn', verbose)
    except KeyboardInterrupt:
        # log('Interrumpting SENDER with KeyboardInterrupt', 'warn', verbose)
        pass
    [obj.close() for obj in [sock_send, led] if obj is not None]
    if is_demo:
        log('Exiting SENDER_RANDOM...', 'info', verbose)
    else:
        log('Exiting ENERPI_LOGGER...', 'info', verbose)


def enerpi_daemon_logger(with_pitemps=False):
    """
    Punto de entrada directa a ENERPI Logger con la configuración de DATA_PATH/config_enerpi.ini.

    Se utiliza para iniciar ENERPI como daemon mediante 'enerpi-daemon start|stop|restart'
    (en conjunción con enerpiweb, con user www-data: 'sudo -u www-data %(path_env_bin)/enerpi-daemon start')

    :param with_pitemps: :bool: Logs RPI temperature every 3 seconds

    """
    sleep(2)
    set_logging_conf(FILE_LOGGING, LOGGING_LEVEL, with_initial_log=False)
    timer_temps = show_pi_temperature(with_pitemps, 3)
    enerpi_logger(path_st=HDF_STORE_PATH, is_demo=False, timeout=None, verbose=False,
                  delta_sampling=SENSORS.delta_sec_data, roll_time=SENSORS.rms_roll_window_sec,
                  sampling_ms=SENSORS.ts_data_ms)
    if timer_temps is not None:
        log('Stopping RPI TEMPS sensing desde enerpi_main_logger...', 'debug', False, True)
        timer_temps.cancel()
