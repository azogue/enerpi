# -*- coding: utf-8 -*-
from collections import deque
import datetime as dt
from flask import Response, request, redirect, url_for, render_template, send_file
import json
import logging
import os
import pandas as pd
from threading import Thread, current_thread
from time import sleep, time

from enerpi.base import BASE_PATH, CONFIG, TZ, DATA_PATH, get_lines_file
from enerpi.api import enerpi_receiver_generator, enerpi_data_catalog
from enerpiplot.plotbokeh import get_bokeh_version, html_plot_buffer_bokeh
from enerpiweb import app, SERVER_FILE_LOGGING, STATIC_PATH, WITH_ML_SUBSYSTEM


ENERPI_FILE_LOGGING = os.path.join(DATA_PATH, CONFIG.get('ENERPI_DATA', 'FILE_LOGGING'))
RSC_GEN_FILE_LOGGING = os.path.join(STATIC_PATH, 'enerpiweb_rscgen.log')
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
    # from enerpiweb.views_labeling import *
    # TODO Integración con las vistas de enerpiprocess (separadas en otro project por dependencias, por ahora)
    pass
# else:
#     @app.route('/learning')
#     def index_learning():
#         return redirect(url_for('control'))


# Interesting files / logs to show:
def _get_filepath_from_file_id(file_id):
    if 'flask' == file_id:
        filename = SERVER_FILE_LOGGING
    elif 'rsc' == file_id:
        filename = RSC_GEN_FILE_LOGGING
    elif 'nginx_err' == file_id:
        filename = '/var/log/nginx/error.log'
    elif 'nginx' == file_id:
        filename = '/var/log/nginx/access.log'
    elif 'enerpi' == file_id:
        filename = ENERPI_FILE_LOGGING
    elif 'uwsgi' == file_id:
        filename = '/var/log/uwsgi/enerpiweb.log'
    else:  # Fichero derivado del catálogo
        cat = enerpi_data_catalog(check_integrity=False)
        if 'raw_store' == file_id:
            filename = os.path.join(cat.base_path, cat.raw_store)
        elif 'catalog' == file_id:
            filename = os.path.join(cat.base_path, cat.catalog_file)
        else:
            logging.error('FILE_ID No reconocido: {}'.format(file_id))
            filename = SERVER_FILE_LOGGING
    return filename


def _format_event_stream(d_msg, timeout_retry=None, msg_id=None):
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


@app.template_filter('ts_strftime')
def ts_strftime(ts):
    try:
        if (ts.hour == 0) and (ts.minute == 0):
            return ts.strftime('%d/%m/%y')
        return ts.strftime('%d/%m/%y %H:%M')
    except AttributeError as e:
        logging.error('AttributeError en template_filter:ts_strftime -> {}'.format(e))
        return str(ts)


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
    # if sys.platform == 'darwin':
    #     sleep(1)
    tic = time()
    if start or end or last_hours:
        cat = enerpi_data_catalog(check_integrity=False)
        if kwh:
            df = cat.get_summary(start=start, end=end, last_hours=last_hours, async_get=False)
        else:
            df = cat.get(start=start, end=end, last_hours=last_hours, async_get=False)
            if (df is not None) and not df.empty:
                df = df[['power', 'ldr', 'ref']]
                df.ldr = df.ldr.astype(float) / 10.
                if last_hours is not None:
                    df_last_data = _get_dataframe_buffer_data()
                    if df_last_data is not None:
                        df_last_data = df_last_data.tz_localize(None)
                        df = pd.DataFrame(pd.concat([df, df_last_data], axis=0)
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
                yield _format_event_stream(send)
                count += 1
                sleep(.5)
            else:
                sleep(.1)
        yield _format_event_stream('CLOSE')

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
            yield _format_event_stream({'success': True, 'table': tabla, 'took': round(toc - toc_p, 3)})
        yield _format_event_stream('CLOSE')

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
            yield _format_event_stream(dict(success=True, b_version=version, script_bokeh=script, bokeh_div=divs[0],
                                            took=round(toc_p - tic, 3)))

            df_print = _get_dataframe_print_buffer_data(df)
            tabla = _get_html_table_buffer_data(df_print)
            toc = time()
            logging.debug('TABLE gen in {:.3f} s'.format(toc - toc_p))
            yield _format_event_stream({'success': True, 'table': tabla, 'took': round(toc - toc_p, 3)})
        yield _format_event_stream('CLOSE')

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


@app.route('/api/hdfstores/<relpath_store>', methods=['GET'])
def download_hdfstore_file(relpath_store=None):
    """
    Devuelve el fichero HDFStore del catálogo de ENERPI pasado como ruta relativa (nombre de fichero .h5).
    * File Download *
    :param relpath_store:
    """
    cat = enerpi_data_catalog(check_integrity=False)
    path_file = cat.get_path_hdf_store_binaries(relpath_store)
    if 'as_attachment' in request.args:
        return send_file(path_file, as_attachment=True, attachment_filename=os.path.basename(path_file))
    return send_file(path_file, as_attachment=False)


@app.route('/api/filedownload/<file_id>', methods=['GET'])
def download_file(file_id):
    """
    * File Download *
    :param file_id:
    """
    filename = _get_filepath_from_file_id(file_id)
    if os.path.exists(filename):
        if 'as_attachment' in request.args:
            return send_file(filename, as_attachment=True, attachment_filename=os.path.basename(filename))
        return send_file(filename, as_attachment=False)
    else:
        msg = json.dumps({'alert_type': 'danger',
                          'texto_alerta': 'El archivo "{}" ({}) no existe!'.format(filename, file_id)})
        return redirect(url_for('control', alerta=msg))


@app.route('/control')
def control():
    """
    Panel de control con links a visores de LOG's, bokeh plots
    y descarga de ficheros de DATA_PATH (para backup remoto)
    """
    global last_data
    last = last_data.copy()
    try:
        is_sender_active = (pd.Timestamp.now(tz=TZ) - last['ts']) < pd.Timedelta('1min')
    except KeyError:
        # last_ts, last_power, host_logger = last_data['ts'], last_data['power'], last_data['host']
        is_sender_active = False
        last = {'host': '?', 'power': -1, 'ts': pd.Timestamp.now(tz=TZ)}
    alerta = request.args.get('alerta', '')
    if alerta:
        alerta = json.loads(alerta)
    cat = enerpi_data_catalog(check_integrity=False)
    df = cat.tree
    if df is not None:
        df = df[df.is_cat & df.is_raw].sort_values(by='ts_ini', ascending=False)
        paths_rel = [(os.path.basename(p), t0.strftime('%d/%m/%y'), tf.strftime('%d/%m/%y'), n)
                     for p, t0, tf, n in zip(df['st'], df['ts_ini'], df['ts_fin'], df['n_rows'])]
    else:
        paths_rel = []
    return render_template('control_panel.html',
                           d_catalog={'path_raw_store': os.path.join(cat.base_path, cat.raw_store),
                                      'path_catalog': os.path.join(cat.base_path, cat.catalog_file),
                                      'ts_init': cat.min_ts, 'ts_catalog': cat.index_ts},
                           d_last_msg=last, is_sender_active=is_sender_active, list_stores=paths_rel, alerta=alerta)


@app.route('/showfile')
@app.route('/showfile/<file>')
def showfile(file='flask'):
    """
    Página de vista de fichero de texto, con orden ascendente / descendente y/o nº de últimas líneas ('tail' de archivo)
    :param file: file_id to show
    """
    delete = request.args.get('delete', '')
    reverse = request.args.get('reverse', False)
    tail_lines = request.args.get('tail', None)

    filename = _get_filepath_from_file_id(file)
    alerta = request.args.get('alerta', '')
    if alerta:
        alerta = json.loads(alerta)
    if not alerta and delete:
        with open(filename, 'w') as f:
            f.close()
        cad_delete = 'LOGFILE {} DELETED'.format(filename.upper())
        logging.warning(cad_delete)
        return redirect(url_for('showfile', file=file,
                                alerta=json.dumps({'alert_type': 'warning', 'texto_alerta': cad_delete})))
    data = get_lines_file(filename, tail=tail_lines, reverse=reverse)
    return render_template('text_file.html', titulo='LOG File:', file_id=file,
                           subtitulo='<strong>{}</strong>'.format(filename),
                           file_content=data, filename=os.path.basename(filename), alerta=alerta)
