# -*- coding: utf-8 -*-
from collections import deque
import datetime as dt
from flask import Response, request, redirect, url_for
import json
import logging
import os
import pandas as pd
from threading import Thread, current_thread
from time import sleep, time

from enerpi import BASE_PATH
from enerpi.api import enerpi_receiver_generator, enerpi_data_catalog
from enerpi.base import CONFIG, TZ, DATA_PATH, get_lines_file
from enerpiplot.plotbokeh import get_bokeh_version, html_plot_buffer_bokeh

from enerpiweb import app, basedir, SERVER_FILE_LOGGING


# WITH_WEB = CONFIG.get('ENERPI_WEBSERVER', 'WITH_WEBSERVER', fallback='False') == 'True'
WITH_ML_SUBSYSTEM = CONFIG.get('ENERPI_WEBSERVER', 'WITH_ML', fallback='False') == 'True'

ENERPI_FILE_LOGGING = CONFIG.get('ENERPI_DATA', 'FILE_LOGGING')
ENERPI_FILE_LOGGING = os.path.join(DATA_PATH, ENERPI_FILE_LOGGING)
PALETA = pd.read_csv(os.path.join(BASE_PATH, 'rsc', 'paleta_power_w.csv')
                     ).set_index('Unnamed: 0')['0'].str[1:-1].str.split(', ').apply(lambda x: [float(i) for i in x])
COLOR_BASE = app.config['BASECOLOR']

BUFFER_MAX_SAMPLES = 1800
STREAM_MAX_TIME = 1800
BOKEH_VERSION = get_bokeh_version()

# GLOBAL VARIABLES FOR ENERPI DATA BROADCAST
last_data = {}
buffer_last_data = deque([], maxlen=BUFFER_MAX_SAMPLES)
thread_receiver = None


if WITH_ML_SUBSYSTEM:
    from enerpiweb.views_labeling import *
else:
    @app.route('/learning')
    def index_learning():
        # TODO Usar redirect a 'control'
        return render_template('control_panel.html')


def format_event_stream(d_msg, timeout_retry=None, msg_id=None):
    if msg_id is not None:
        return 'id: {}\ndata: {}\n\n'.format(msg_id, json.dumps(d_msg))
    if timeout_retry is not None:
        return 'retry: {}\ndata: {}\n\n'.format(timeout_retry, json.dumps(d_msg))
    return 'data: {}\n\n'.format(json.dumps(d_msg))


def _aplica_paleta(serie):
    return ['background-color: rgba({}, {}, {}, .7); color: #fff'.format(
        *map(lambda x: int(255 * x), PALETA.loc[:v].iloc[-1])) for v in serie]


@app.template_filter('text_date')
def text_date(str_date):
    try:
        delta = dt.timedelta(days=int(str_date))
        return (dt.date.today() + delta).strftime('%Y-%m-%d')
    except ValueError:
        if str_date == 'today':
            return dt.date.today().strftime('%Y-%m-%d')
        elif str_date == 'yesterday':
            return (dt.date.today() - dt.timedelta(days=1)).strftime('%Y-%m-%d')
        return dt.date.today().strftime('Err_%Y-%m-%d')


def _get_dataframe_buffer_data():
    global buffer_last_data
    if len(buffer_last_data) > 0:
        df = pd.DataFrame(list(buffer_last_data))[['ts', 'power', 'ldr', 'ref']].set_index('ts')
        df.power = df.power.astype(int)
        df.ref = df.ref.astype(int)
        df.ldr = pd.Series(df.ldr * 100).round(1)
        return df
    return None


def _get_dataframe_print_buffer_data(df=None):
    if df is None:
        df = _get_dataframe_buffer_data()
    df_print = df.rename(columns={'ref': 'nº s.', 'power': 'Power (W)', 'ldr': 'LDR (%)'})
    df_print['T'] = df_print.index.map(lambda x: x.time().strftime('%H:%M:%S'))
    df_print = df_print.reset_index(drop=True)[['T', 'Power (W)', 'LDR (%)', 'nº s.']].set_index('T')
    return df_print


def _get_html_table_buffer_data(df_print=None):
    if df_print is None:
        df_print = _get_dataframe_print_buffer_data().iloc[::-1]
    clases_tabla = 'table table-responsive table-stripped table-hover table-not-bordered table-sm'
    tabla = (df_print.style  # .set_uuid('idunico-tablebuffer')
             .set_table_attributes('class="{}" border="0"'.format(clases_tabla))
             .apply(_aplica_paleta, subset=['Power (W)'])
             .format({'LDR (%)': lambda x: "{:.1f} %".format(x),
                      'Power (W)': lambda x: "<strong>{:.0f}</strong> W".format(x)})
             # .bar(subset=['LDR (%)'], color='yellow')
             .set_properties(**{'font-size': '12pt', 'font-family': 'Calibri', 'text-align': 'center'})
             .render())
    return tabla


def _gen_stream_data_bokeh(start=None, end=None, last_hours=None, rs_data=None, rm_data=None,
                           use_median=False, kwh=False):
    tic = time()
    if start or end or last_hours:
        cat = enerpi_data_catalog(check_integrity=False)
        if kwh:
            df = cat.get_summary(start=start, end=end, last_hours=last_hours, async_get=False)
        else:
            df = cat.get(start=start, end=end, last_hours=last_hours, async_get=False)[['power', 'ldr', 'ref']]
            df.ldr = df.ldr.astype(float) / 10.
            if last_hours is not None:
                df = pd.DataFrame(pd.concat([df, _get_dataframe_buffer_data().tz_localize(None)], axis=0)
                                  ).sort_index().drop_duplicates(keep='last')
            df = cat.resample_data(df, rs_data=rs_data, rm_data=rm_data, use_median=use_median)
    else:
        df = _get_dataframe_buffer_data()
    toc_df = time()
    if df is not None and not df.empty:
        try:
            script, divs, version = html_plot_buffer_bokeh(df, is_kwh_plot=kwh)
            toc_p = time()
            logging.debug('Bokeh plot gen in {:.3f} s; pd.df in {:.3f} s.'.format(toc_p - toc_df, toc_df - tic))
            yield format_event_stream(dict(success=True,
                                           b_version=version, script_bokeh=script, bokeh_div=divs[0],
                                           took=round(toc_p - tic, 3), took_df=round(toc_df - tic, 3)))
        except Exception as e:
            print('ERROR: ', e, e.__class__)
            yield format_event_stream(dict(success=False, error=e))
    else:
        yield format_event_stream(dict(success=False))
    yield format_event_stream('CLOSE')


###############################
# BROADCAST RECEIVER
###############################
def _init_receiver():
    logging.info('**INIT_BROADCAST_RECEIVER en PID={}'.format(os.getpid()))
    gen = enerpi_receiver_generator()
    count = 0
    while True:
        try:
            last = next(gen)
            global last_data
            global buffer_last_data
            last_data = last.copy()
            buffer_last_data.append(last_data)
        except StopIteration:
            logging.debug('StopIteration on counter={}'.format(count))
            if count > 0:
                sleep(2)
                gen = enerpi_receiver_generator()
        count += 1
        sleep(.5)
    logging.warning('**BROADCAST_RECEIVER en PID={}, thread={}. CLOSED on counter={}'
                    .format(os.getpid(), current_thread(), count))


@app.before_first_request
def init_receiver():
    global thread_receiver
    # Broadcast receiver
    if not thread_receiver:
        thread_receiver = Thread(target=_init_receiver)
        thread_receiver.setDaemon(True)
        thread_receiver.start()


###############################
# STREAM REAL-TIME SENSORS DATA
###############################
@app.route("/api/stream/realtime", methods=["GET"])
def stream_sensors():

    def _gen_stream_last_values():
        last_ts = dt.datetime.now(tz=TZ) - dt.timedelta(minutes=1)
        count = 0
        tic = time()
        while time() - tic < STREAM_MAX_TIME:
            global last_data
            if 'ts' in last_data and last_data['ts'] > last_ts:
                send = last_data.copy()
                last_ts = send['ts']
                send['ts'] = send['ts'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
                yield format_event_stream(send)
                count += 1
                sleep(.5)
            else:
                sleep(.1)
        yield format_event_stream('CLOSE')

    # print('request stream_sensors Accept-Language:', request.headers['Accept-Language'])
    return Response(_gen_stream_last_values(), mimetype='text/event-stream')


#############################
# INDEX & ROUTES
#############################
@app.route('/monitor')
def only_tiles():
    ldr = request.args.get('ldr', 'true') == 'true'
    consumption = request.args.get('consumption', 'false') != 'false'
    return render_template('only_tiles.html', include_consumption=consumption, include_ldr=ldr)


@app.route('/')
@app.route('/index')
def index():
    global buffer_last_data
    return render_template('index.html', include_consumption=True, include_ldr=True,
                           last_samples=list(buffer_last_data))


@app.route('/api/stream/table')
def table_buffer():

    def _gen_stream_data_table():
        df = _get_dataframe_buffer_data()
        if df is not None:
            df_print = _get_dataframe_print_buffer_data(df)
            toc_p = time()
            tabla = _get_html_table_buffer_data(df_print)
            toc = time()
            logging.debug('TABLE gen in {:.3f} s'.format(toc - toc_p))
            yield format_event_stream({'success': True, 'table': tabla, 'took': round(toc - toc_p, 3)})
        yield format_event_stream('CLOSE')

    return Response(_gen_stream_data_table(), mimetype='text/event-stream')


@app.route('/api/stream/bokehtable')
def bokeh_table_buffer():

    def _gen_stream_data_table_bokeh():
        tic = time()
        df = _get_dataframe_buffer_data()
        toc_df = time()
        if df is not None:
            script, divs, version = html_plot_buffer_bokeh(df)
            toc_p = time()
            logging.debug('Bokeh plot gen in {:.3f} s; pd.df in {:.3f} s.'.format(toc_p - toc_df, toc_df - tic))
            yield format_event_stream(dict(success=True,
                                           b_version=version, script_bokeh=script, bokeh_div=divs[0],
                                           took=round(toc_p - tic, 3)))

            df_print = _get_dataframe_print_buffer_data(df)
            tabla = _get_html_table_buffer_data(df_print)
            toc = time()
            logging.debug('TABLE gen in {:.3f} s'.format(toc - toc_p))
            yield format_event_stream({'success': True, 'table': tabla, 'took': round(toc - toc_p, 3)})
        yield format_event_stream('CLOSE')

    return Response(_gen_stream_data_table_bokeh(), mimetype='text/event-stream')


@app.route('/api/stream/bokeh', methods=['GET'])
@app.route('/api/stream/bokeh/from/<start>', methods=['GET'])
@app.route('/api/stream/bokeh/from/<start>/to/<end>', methods=['GET'])
@app.route('/api/stream/bokeh/last/<last_hours>', methods=['GET'])
def bokeh_buffer(start=None, end=None, last_hours=None):
    # print(request.url)
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


@app.route('/bokehplot')
def bokehplot():
    return render_template('bokeh_plot.html', url_stream_bokeh=url_for('bokeh_buffer'), b_version=BOKEH_VERSION)


@app.route('/tablelast')
def tablelast():
    return render_template('table_buffer.html', url_table=url_for('table_buffer'))


@app.route('/control')
def control():
    return render_template('control_panel.html')


@app.route('/showfile')
@app.route('/showfile/<file>')
def showfile(file='flask'):
    alerta = request.args.get('alerta', '')
    delete = request.args.get('delete', '')
    reverse = request.args.get('reverse', False)
    tail_lines = request.args.get('tail', None)

    select_log = file
    if 'flask' in select_log:
        filename = SERVER_FILE_LOGGING
    elif 'rsc' in select_log:
        filename = os.path.join(basedir, 'static', 'enerpiweb_rscgen.log')
    elif 'nginx_err' in select_log:
        filename = '/var/log/nginx/error.log'
    elif 'nginx' in select_log:
        filename = '/var/log/nginx/access.log'
    elif 'enerpi' in select_log:
        filename = ENERPI_FILE_LOGGING
    elif 'uwsgi' in select_log:
        filename = '/var/log/uwsgi/enerpiweb.log'
    else:
        filename = SERVER_FILE_LOGGING

    if alerta:
        alerta = json.loads(alerta)
    if not alerta and delete:
        with open(filename, 'w') as f:
            f.close()
        cad_delete = 'LOGFILE {} DELETED'.format(filename.upper())
        logging.warning(cad_delete)
        return redirect(url_for('showfile', file=select_log, alerta=json.dumps({'alert_type': 'warning',
                                                                                'texto_alerta': cad_delete})))
    data = get_lines_file(filename, tail=tail_lines, reverse=reverse)
    return render_template('text_file.html', titulo='LOG File:', file_id=select_log,
                           subtitulo='<strong>{}</strong>'.format(filename),
                           file_content=data, filename=os.path.basename(filename), alerta=alerta)
