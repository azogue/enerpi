# -*- coding: utf-8 -*-
"""
Flask routes for streamed ENERPI content
    - Real-time enerpi values (receiving the ENERPI broadcast)
    - Bokeh plots
        - from buffered data (from ENERPI broadcast)
        - from catalog data (combined with ENERPI broadcast data)
    - Last broadcasted value / boolean 'stream_is_alive'
    etc...

"""
from collections import deque
import datetime as dt
from flask import Response, request, render_template, jsonify
import json
import os
import pandas as pd
from threading import Thread, Event, current_thread
from time import sleep, time

from enerpi.base import SENSORS, log
from enerpi.api import enerpi_receiver_generator, enerpi_data_catalog
from enerpiplot.plotbokeh import get_bokeh_version, html_plot_buffer_bokeh
from enerpiweb import app, auto


BUFFER_MAX_SAMPLES = 1800
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
        is_sender_active = (pd.Timestamp.now(tz=SENSORS.TZ) - last[SENSORS.ts_column]) < pd.Timedelta('5s')
    except KeyError:
        is_sender_active = False
        last = {'host': '?', SENSORS.main_column: -1, SENSORS.ts_column: pd.Timestamp.now(tz=SENSORS.TZ)}
    return is_sender_active, last


def _format_event_stream(d_msg):
    return 'data: {}\n\n'.format(json.dumps(d_msg))


def _get_dataframe_buffer_data():
    global buffer_last_data
    if len(buffer_last_data) > 0:
        df = pd.DataFrame(list(buffer_last_data))
        try:
            df = df[SENSORS.columns_sampling].set_index(SENSORS.ts_column)
        except KeyError as e:
            log('KeyError "{}" in _get_dataframe_buffer_data'.format(e), 'error', True)
            df = df[list(filter(lambda x: x in df, SENSORS.columns_sampling))].set_index(SENSORS.ts_column)
        for s in filter(lambda x: x.name in df, SENSORS):
            if s.is_rms:
                df[s.name] = df[s.name].round(0).astype('float32')
            else:
                df[s.name] = pd.Series(df[s.name] * 1000).round(0).astype('int16')
        for c in filter(lambda x: x in df, [SENSORS.ref_rms, SENSORS.ref_mean]):
            if c in df:
                df[c] = df[c].astype('int16')
        return df
    return None


def _gen_stream_data_bokeh(start=None, end=None, last_hours=None,
                           rs_data=None, use_median=False, kwh=False):
    tic = time()
    if start or end or last_hours:
        cat = enerpi_data_catalog(check_integrity=False)
        if kwh:
            df = cat.get_summary(start=start, end=end, last_hours=last_hours)
        else:
            df = cat.get(start=start, end=end, last_hours=last_hours)
            if (df is not None) and not df.empty:
                if last_hours is not None:
                    df_last_data = _get_dataframe_buffer_data()
                    if df_last_data is not None:
                        df_last_data = df_last_data.tz_localize(None)
                        df = pd.DataFrame(pd.concat([df, df_last_data])
                                          ).sort_index().drop_duplicates(keep='last')
                df = cat.resample_data(df, rs_data=rs_data, use_median=use_median)
    else:
        df = _get_dataframe_buffer_data()
    toc_df = time()
    if df is not None and not df.empty:
        script, divs, version = html_plot_buffer_bokeh(df, is_kwh_plot=kwh)
        toc_p = time()
        log('Bokeh plot gen in {:.3f} s; pd.df in {:.3f} s.'.format(toc_p - toc_df, toc_df - tic), 'debug', False)
        yield _format_event_stream(dict(success=True, b_version=version, script_bokeh=script, bokeh_div=divs[0],
                                        took=round(toc_p - tic, 3), took_df=round(toc_df - tic, 3)))
    else:
        msg = ('No data for BOKEH PLOT: start={}, end={}, last_hours={}, '
               'rs_data={}, use_median={}, kwh={}<br>--> DATA: {}'
               .format(start, end, last_hours, rs_data, use_median, kwh, df))
        yield _format_event_stream(dict(success=False, error=msg))
    yield _format_event_stream('CLOSE')


###############################
# BROADCAST RECEIVER
###############################
class ReceiverThread(Thread):
    """
    Thread for receiving real-time broadcast values from ENERPI LOGGER
    """
    def __init__(self):
        self._stopevent = Event()
        Thread.__init__(self, name='ReceiverThread')

    def run(self):
        """ main control loop """
        log('**INIT_BROADCAST_RECEIVER en PID={}'.format(os.getpid()), 'info', False)
        gen = enerpi_receiver_generator()
        count = 0
        while not self._stopevent.isSet():
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
        thread_receiver = ReceiverThread()
        thread_receiver.setDaemon(True)
        thread_receiver.start()


###############################
# STREAM REAL-TIME SENSORS DATA
###############################
@app.route('/api/last', methods=['GET'])
@auto.doc()
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
@auto.doc()
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
        stream_max_time = app.config['STREAM_MAX_TIME']
        while time() - tic < stream_max_time:
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
@auto.doc()
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
    if 'use_median' in request.args:
        kwargs.update(use_median=request.args['use_median'])
    return Response(_gen_stream_data_bokeh(**kwargs), mimetype='text/event-stream')


###############################
# INDEX WITH REAL TIME MONITOR
###############################
@app.route('/index', methods=['GET'])
@auto.doc()
def index():
    """
    Webserver index page with real time monitoring of ENERPI

    """
    global buffer_last_data
    with_last_samples = 'table' in request.args
    data_monitor = SENSORS.to_dict()
    data_monitor.update(ts=SENSORS.ts_column)
    return render_template('index.html', include_consumption=True, data_monitor=data_monitor,
                           with_last_samples=with_last_samples, last_samples=list(buffer_last_data))


@app.route('/api/monitor', methods=['GET'])
@auto.doc()
def only_tiles():
    """
    Url route for showing only the real-time monitoring tiles
    (for emmbed in another place, e.g.)

    """
    consumption = request.args.get('consumption', 'false') != 'false'
    data_monitor = SENSORS.to_dict()
    data_monitor.update(ts=SENSORS.ts_column)
    return render_template('only_tiles.html', include_consumption=consumption, data_monitor=data_monitor)
