# -*- coding: utf-8 -*-
from collections import deque
import datetime as dt
from flask import render_template, Response, request, redirect, url_for
import json
import logging
from threading import Thread, current_thread
from time import sleep, time
import os
import pytz
from enerpiweb import app, basedir, SERVER_FILE_LOGGING
from enerpi import BASE_PATH
from enerpi.base import CONFIG, get_lines_file
from enerpi.api import enerpi_receiver_generator
from enerpi.plotbokeh import get_bokeh_version, html_plot_buffer_bokeh
import pandas as pd

TZ = pytz.timezone(CONFIG.get('ENERPI_SAMPLER', 'TZ', fallback='Europe/Madrid'))
DATA_PATH = CONFIG.get('ENERPI_DATA', 'DATA_PATH')
ENERPI_FILE_LOGGING = CONFIG.get('ENERPI_DATA', 'FILE_LOGGING')
ENERPI_FILE_LOGGING = os.path.join(DATA_PATH, ENERPI_FILE_LOGGING)
PALETA = pd.read_csv(os.path.join(BASE_PATH, 'rsc', 'paleta_power_w.csv')
                     ).set_index('Unnamed: 0')['0'].str[1:-1].str.split(', ').apply(lambda x: [float(i) for i in x])
COLOR_BASE = app.config['BASECOLOR']
STREAM_MAX_TIME = 1800
BOKEH_VERSION = get_bokeh_version()

# GLOBAL VARIABLES FOR ENERPI DATA BROADCAST
last_data = {}
buffer_last_data = deque([], maxlen=600)
thread_receiver = None


def format_event_stream(d_msg, timeout_retry=None, msg_id=None):
    if msg_id is not None:
        return 'id: {}\ndata: {}\n\n'.format(msg_id, json.dumps(d_msg))
    if timeout_retry is not None:
        return 'retry: {}\ndata: {}\n\n'.format(timeout_retry, json.dumps(d_msg))
    return 'data: {}\n\n'.format(json.dumps(d_msg))


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
            # if count < 2:
            #     logging.debug(' 1º MSG RECEIVED: {}'.format(last_data))
        except StopIteration:
            logging.debug('StopIteration on counter={}'.format(count))
            if count > 0:
                sleep(2)
                gen = enerpi_receiver_generator()
            # elif count > 100:
            #     logging.debug('Exiting receiver on counter={}'.format(count))
            #     break
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
@app.route("/stream_enerpi", methods=["GET"])
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
        # print('CLOSING STREAM ENERPI')
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


@app.route('/api/tablebuffer')
def table_buffer():

    def _aplica_paleta(serie):
        return ['background-color: rgba({}, {}, {}, .7); color: #fff'.format(
            *map(lambda x: int(255 * x), PALETA.loc[:v].iloc[-1])) for v in serie]

    def _gen_stream_data_table():
        tic, tabla = time(), None
        global buffer_last_data
        if len(buffer_last_data) > 0:
            df = pd.DataFrame(list(buffer_last_data))[['ts', 'power', 'ldr', 'ref']].set_index('ts').iloc[::-1]
            df.power = df.power.astype(int)
            df.ref = df.ref.astype(int)
            df.ldr = pd.Series(df.ldr * 100).round(1)
            toc_df = time()

            script, divs, version = html_plot_buffer_bokeh(df.iloc[::-1], COLOR_BASE)
            toc_p = time()
            logging.debug('Bokeh plot gen in {:.3f} s; pd.df in {:.3f} s.'.format(toc_p - toc_df, toc_df - tic))
            yield format_event_stream(dict(success=True,
                                           b_version=version, script_bokeh=script, bokeh_div=divs[0],
                                           took=round(toc_p - tic, 3)))

            df_print = df.copy().rename(columns={'ref': 'nº s.', 'power': 'Power (W)', 'ldr': 'LDR (%)'})
            df_print['T'] = df_print.index.map(lambda x: x.time().strftime('%H:%M:%S'))
            df_print = df_print.reset_index(drop=True)[['T', 'Power (W)', 'LDR (%)', 'nº s.']].set_index('T')

            clases_tabla = 'table table-responsive table-stripped table-hover table-not-bordered table-sm'
            tabla = (df_print.style  # .set_uuid('idunico-tablebuffer')
                     .set_table_attributes('class="{}" border="0"'.format(clases_tabla))
                     .apply(_aplica_paleta, subset=['Power (W)'])
                     .format({'LDR (%)': lambda x: "{:.1f} %".format(x),
                              'Power (W)': lambda x: "<strong>{:.0f}</strong> W".format(x)})
                     # .bar(subset=['LDR (%)'], color='yellow')
                     .set_properties(**{'font-size': '12pt', 'font-family': 'Calibri', 'text-align': 'center'})
                     .render())
            toc = time()
            logging.debug('TABLE gen in {:.3f} s'.format(toc - toc_p))
            yield format_event_stream({'success': True, 'table': tabla, 'took': round(toc - toc_p, 3)})
        yield format_event_stream('CLOSE')

    return Response(_gen_stream_data_table(), mimetype='text/event-stream')


@app.route('/control')
def control():
    return render_template('control_panel.html',
                           url_table=url_for('table_buffer'), b_version=BOKEH_VERSION)


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
