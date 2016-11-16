# -*- coding: utf-8 -*-
from collections import deque
import datetime as dt
from flask import Response, request, render_template, jsonify
import json
import os
import pandas as pd
from threading import Thread, current_thread
from time import sleep, time

from enerpi.base import SENSORS, log
from enerpi.api import enerpi_receiver_generator, enerpi_data_catalog
from enerpiplot.plotbokeh import get_bokeh_version, html_plot_buffer_bokeh
from enerpiweb import app


BUFFER_MAX_SAMPLES = 1800
STREAM_MAX_TIME = 1800
BOKEH_VERSION = get_bokeh_version()

# GLOBAL VARIABLES FOR ENERPI DATA BROADCAST
last_data = {}
buffer_last_data = deque([], maxlen=BUFFER_MAX_SAMPLES)
thread_receiver = None


def stream_is_alive():
    """
    Check enerpi emitter stream
    """
    global last_data
    last = last_data.copy()
    try:
        is_sender_active = (pd.Timestamp.now(tz=SENSORS.TZ) - last[SENSORS.ts_column]) < pd.Timedelta('1min')
    except KeyError:
        # last_ts, last_power, host_logger = last_data['ts'], last_data['power'], last_data['host']
        is_sender_active = False
        last = {'host': '?', SENSORS.main_column: -1, SENSORS.ts_column: pd.Timestamp.now(tz=SENSORS.TZ)}
    return is_sender_active, last


def _format_event_stream(d_msg, timeout_retry=None, msg_id=None):
    if msg_id is not None:
        return 'id: {}\ndata: {}\n\n'.format(msg_id, json.dumps(d_msg))
    if timeout_retry is not None:
        return 'retry: {}\ndata: {}\n\n'.format(timeout_retry, json.dumps(d_msg))
    return 'data: {}\n\n'.format(json.dumps(d_msg))


def _get_dataframe_buffer_data():
    global buffer_last_data
    if len(buffer_last_data) > 0:
        df = pd.DataFrame(list(buffer_last_data))
        try:
            df = df[SENSORS.columns_sampling].set_index(SENSORS.ts_column)
        except KeyError as e:
            log('KeyError "{}" in _get_dataframe_buffer_data'.format(e), 'error')
            df = df[list(filter(lambda x: x in df, SENSORS.columns_sampling))].set_index(SENSORS.ts_column)
        for s in filter(lambda x: x.name in df, SENSORS):
            if s.is_rms:
                df[s.name] = df[s.name].astype(int)
            else:
                df[s.name] = pd.Series(df[s.name] * 100).round(1)
        for c in filter(lambda x: x in df, [SENSORS.ref_rms, SENSORS.ref_mean]):
            if c in df:
                try:
                    df[c] = df[c].astype(int)
                except ValueError:
                    df[c] = df[c].astype(float).astype(int)
        return df
    return None


def _gen_stream_data_bokeh(start=None, end=None, last_hours=None,
                           rs_data=None, rm_data=None, use_median=False, kwh=False):
    tic = time()
    if start or end or last_hours:
        cat = enerpi_data_catalog(check_integrity=False)
        if kwh:
            df = cat.get_summary(start=start, end=end, last_hours=last_hours)
        else:
            df = cat.get(start=start, end=end, last_hours=last_hours)
            if (df is not None) and not df.empty:
                df = df[SENSORS.columns_sampling]
                # TODO Revisar formatos de analog mean!
                # SENSORS
                # df.ldr = df['ldr'].astype(float) / 10.
                if last_hours is not None:
                    df_last_data = _get_dataframe_buffer_data()
                    if df_last_data is not None:
                        df_last_data = df_last_data.tz_localize(None)
                        df = pd.DataFrame(pd.concat([df, df_last_data])
                                          ).sort_index().drop_duplicates(keep='last')
                df = cat.resample_data(df, rs_data=rs_data, rm_data=rm_data, use_median=use_median)
    else:
        df = _get_dataframe_buffer_data()
    toc_df = time()
    if df is not None and not df.empty:
        try:
            script, divs, version = html_plot_buffer_bokeh(df, is_kwh_plot=kwh)
            toc_p = time()
            log('Bokeh plot gen in {:.3f} s; pd.df in {:.3f} s.'.format(toc_p - toc_df, toc_df - tic), 'debug', False)
            yield _format_event_stream(dict(success=True, b_version=version, script_bokeh=script, bokeh_div=divs[0],
                                            took=round(toc_p - tic, 3), took_df=round(toc_df - tic, 3)))
        except Exception as e:
            msg = 'ERROR en: BOKEH PLOT: {} [{}]'.format(e, e.__class__)
            print(msg)
            yield _format_event_stream(dict(success=False, error=msg))
    else:
        msg = ('No hay datos para BOKEH PLOT: start={}, end={}, last_hours={}, '
               'rs_data={}, rm_data={}, use_median={}, kwh={}<br>--> DATA: {}'
               .format(start, end, last_hours, rs_data, rm_data, use_median, kwh, df))
        # print(msg.replace('<br>', '\n'))
        yield _format_event_stream(dict(success=False, error=msg))
    yield _format_event_stream('CLOSE')


###############################
# BROADCAST RECEIVER
###############################
def _init_receiver_thread():
    log('**INIT_BROADCAST_RECEIVER en PID={}'.format(os.getpid()), 'info', False)
    gen = enerpi_receiver_generator()
    count = 0
    while True:
        try:
            last = next(gen)
            global last_data
            global buffer_last_data
            last_data = last.copy()
            buffer_last_data.append(last_data)
            # print('DEBUG STREAM: last_data --> ', last_data)
        except StopIteration:
            log('StopIteration on counter={}'.format(count), 'debug', False)
            if count > 0:
                sleep(2)
                gen = enerpi_receiver_generator()
            else:
                log('Not receiving broadcast msgs. StopIteration at init!', 'error', False)
                break
        count += 1
        sleep(.5)
    log('**BROADCAST_RECEIVER en PID={}, thread={}. CLOSED on counter={}'
        .format(os.getpid(), current_thread(), count), 'warn', False)


@app.before_first_request
def _init_receiver():
    global thread_receiver
    # Broadcast receiver
    if not thread_receiver:
        thread_receiver = Thread(target=_init_receiver_thread)
        thread_receiver.setDaemon(True)
        thread_receiver.start()


###############################
# STREAM REAL-TIME SENSORS DATA
###############################
@app.route('/api/last')
def last_value_json():
    """
    Return last received msg from ENERPI logger in JSON

    :return: :json: msg
    """
    global last_data
    send = last_data.copy()
    if SENSORS.ts_column in send:
        send[SENSORS.ts_column] = send[SENSORS.ts_column].strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
    return jsonify(send)


@app.route("/api/stream/realtime", methods=["GET"])
def stream_sensors():
    """
    Stream real-time data as it is received from ENERPI broadcast.
    Used for real-time monitoring of ENERPI values.

    :return: SSE stream response

    """
    def _gen_stream_last_values():
        last_ts = dt.datetime.now(tz=SENSORS.TZ) - dt.timedelta(minutes=1)
        count = 0
        tic = time()
        while time() - tic < STREAM_MAX_TIME:
            global last_data
            if SENSORS.ts_column in last_data and last_data[SENSORS.ts_column] > last_ts:
                send = last_data.copy()
                last_ts = send[SENSORS.ts_column]
                send[SENSORS.ts_column] = send[SENSORS.ts_column].strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
                yield _format_event_stream(send)
                count += 1
                sleep(.5)
            else:
                sleep(.1)
        yield _format_event_stream('CLOSE')

    return Response(_gen_stream_last_values(), mimetype='text/event-stream')


@app.route('/api/stream/bokeh', methods=['GET'])
@app.route('/api/stream/bokeh/from/<start>', methods=['GET'])
@app.route('/api/stream/bokeh/from/<start>/to/<end>', methods=['GET'])
@app.route('/api/stream/bokeh/last/<last_hours>', methods=['GET'])
def bokeh_buffer(start=None, end=None, last_hours=None):
    """
    Stream the bokeh data to make a plot.
    Used for load/reload plots from user queries.

    :param start: :str: start datetime of data query
    :param end: :str: end datetime of data query
    :param last_hours: :str: query from 'last hours' until now

    :return: SSE stream response

    """
    kwargs = dict(start=start, end=end, last_hours=last_hours)
    if 'kwh' in request.args:
        kwargs.update(kwh=request.args['kwh'] == 'true')
    if 'rs_data' in request.args:
        kwargs.update(rs_data=request.args['rs_data'])
    if 'rm_data' in request.args:
        kwargs.update(rm_data=request.args['rm_data'])
    if 'use_median' in request.args:
        kwargs.update(use_median=request.args['use_median'])
    return Response(_gen_stream_data_bokeh(**kwargs), mimetype='text/event-stream')


###############################
# INDEX WITH REAL TIME MONITOR
###############################
@app.route('/index')
def index():
    """
    Webserver index page with real time monitoring of ENERPI

    """
    global buffer_last_data
    with_last_samples = 'table' in request.args
    return render_template('index.html', include_consumption=True, include_ldr=True,
                           with_last_samples=with_last_samples, last_samples=list(buffer_last_data))
