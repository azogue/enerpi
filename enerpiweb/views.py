# -*- coding: utf-8 -*-
"""
Flask main routes for ENERPIWEB

"""
import datetime as dt
from flask import request, redirect, url_for, render_template, jsonify, abort
import json
from threading import Timer
import os
from pandas import TimeGrouper
import sys
from enerpi.base import log, SENSORS
from enerpi.api import enerpi_data_catalog
from enerpiplot.plotbokeh import get_bokeh_version, COLOR_REF_RMS, COLS_DATA_KWH, COLORS_DATA_KWH
from enerpiweb import app, auto, WITH_ML_SUBSYSTEM
from enerpiweb.rt_stream import stream_is_alive
from enerpiweb.forms import DummyForm


BOKEH_VERSION = get_bokeh_version()

if WITH_ML_SUBSYSTEM:
    # from enerpiweb.views_labeling import *
    # TODO Integración con las vistas de enerpiprocess (separadas en otro project por dependencias, por ahora)
    pass
# else:
#     @app.route('/learning')
#     def index_learning():
#         return redirect(url_for('control'))


#############################
# INDEX & ROUTES
#############################
@app.route('/control', methods=['GET'])
@auto.doc()
def control():
    """
    Admin Control Panel with links to LOG viewing/downloading, hdf_stores download, ENERPI config editor, etc.

    """
    def _text_button_hdfstore(rel_path, ini, fin, nrows):
        name = os.path.basename(os.path.splitext(rel_path)[0])
        ini, fin = ini.strftime('%d/%m/%y'), fin.strftime('%d/%m/%y')
        if ini == fin:
            t_date = ini
        else:
            t_date = '{}->{}'.format(ini, fin)
        text = '''<i class="fa fa-file-archive-o" aria-hidden="true"></i><strong> {}</strong><br>   '''.format(name)
        text += '''<small>({}, N={})</small>'''.format(t_date, nrows)
        return text

    is_sender_active, last = stream_is_alive()
    after_sysop = request.args.get('after_sysop', '')
    alerta = request.args.get('alerta', '')
    if alerta:
        alerta = json.loads(alerta)
    cat = enerpi_data_catalog(check_integrity=False)
    df = cat.tree
    if df is not None:
        df = df[df.is_cat & df.is_raw].sort_values(by='ts_ini', ascending=False)
        paths_rel = [(os.path.basename(p), _text_button_hdfstore(p, t0, tf, n))
                     for p, t0, tf, n in zip(df['st'], df['ts_ini'], df['ts_fin'], df['n_rows'])]
    else:
        paths_rel = []
    form_operate = DummyForm()
    return render_template('control_panel.html',
                           d_catalog={'path_raw_store': os.path.join(cat.base_path, cat.raw_store),
                                      'path_catalog': os.path.join(cat.base_path, cat.catalog_file),
                                      'ts_init': cat.min_ts, 'ts_catalog': cat.index_ts},
                           d_last_msg=last, is_sender_active=is_sender_active, list_stores=paths_rel,
                           form_operate=form_operate, after_sysop=after_sysop,
                           alerta=alerta)


@app.route('/api/help', methods=['GET'])
@auto.doc()
def api_help():
    """
    Documentation page generated with 'Autodoc', with custom template;
    or json response with server routes (with ?json=true)

    """
    w_json = request.args.get('json', False)
    if w_json:
        endpoints = [rule.rule for rule in app.url_map.iter_rules()
                     if rule.endpoint != 'static']
        return jsonify(dict(api_endpoints=endpoints))
    return auto.html(template='doc/api_help.html')


@app.route('/api/bokehplot', methods=['GET'])
@auto.doc()
def bokehplot():
    """
    Base webpage for query & show bokeh plots of ENERPI data

    """
    cols_disp = SENSORS.columns_sensors_rms + SENSORS.columns_sensors_mean
    colors_vars = {c: SENSORS[c].color for c in cols_disp}
    checked = {c: True for c in cols_disp}
    cols_disp += [SENSORS.ref_rms]
    colors_vars.update({SENSORS.ref_rms: COLOR_REF_RMS})
    checked.update({SENSORS.ref_rms: False})
    cols_disp_arg = ','.join(cols_disp[:-1])
    checked_kwh = {COLS_DATA_KWH[0]: True, COLS_DATA_KWH[1]: False, COLS_DATA_KWH[2]: False}
    return render_template('bokeh_plot.html', url_stream_bokeh=url_for('bokeh_buffer', columns=cols_disp_arg),
                           b_version=BOKEH_VERSION, columns=cols_disp, colors_vars=colors_vars,
                           columns_kwh=COLS_DATA_KWH[:-1], checked=checked, checked_kwh=checked_kwh,
                           colors_vars_kwh={c: color for c, color in zip(COLS_DATA_KWH[:-1], COLORS_DATA_KWH)})


@app.route('/', methods=['GET'])
@auto.doc()
def base_index():
    """
    Redirects to 'index', with real-time monitoring tiles of ENERPI sensors

    """
    return redirect(url_for('index'))


#############################
# DATA API
#############################
def _get_enerpi_data(start=None, end=None, is_consumption=True):
    if not (start or end):
        start = (dt.datetime.now(tz=SENSORS.TZ) - dt.timedelta(days=7)
                 ).replace(hour=0, minute=0, second=0, microsecond=0)  # 1 week
    else:
        if start and (type(start) is str):
            start = start.replace('_', ' ')
        if end and (type(end) is str):
            end = end.replace('_', ' ')
    cat = enerpi_data_catalog(check_integrity=False)
    if is_consumption:
        df = cat.get_summary(start=start, end=end)
    else:
        df = cat.get(start=start, end=end, column=SENSORS.main_column)
    return df


@app.route('/api/consumption/from/<start>', methods=['GET'])
@app.route('/api/consumption/from/<start>/to/<end>', methods=['GET'])
@auto.doc()
def consumption_data(start, end=None):
    """
    Endpoint for get consumption data from enerPI.

    :param start: start time of data interval
    :param end: end time of data interval

    """
    data = _get_enerpi_data(start, end, is_consumption=True)
    if (data is not None) and not data.empty and ('kWh' in data):
        daily_sum = request.args.get('daily', 'False').lower() == 'true'
        round_prec = int(request.args.get('round', '4'))
        consumption = data['kWh']
        consumption.index = consumption.index.tz_localize(SENSORS.TZ)
        if daily_sum:
            consumption = consumption.groupby(TimeGrouper('D')).sum()
        return jsonify(json.loads(consumption.to_json(double_precision=round_prec)))
    return abort(500)


@app.route('/api/power/from/<start>', methods=['GET'])
@app.route('/api/power/from/<start>/to/<end>', methods=['GET'])
@auto.doc()
def mainpower_data(start, end=None):
    """
    Endpoint for get main power raw data from enerPI.

    :param start: start time of data interval
    :param end: end time of data interval

    """
    data = _get_enerpi_data(start, end, is_consumption=False)
    if (data is not None) and not data.empty:
        daily_sum = request.args.get('daily', 'False').lower() == 'true'
        round_prec = int(request.args.get('round', '4'))
        data.index = data.index.tz_localize(SENSORS.TZ)
        if daily_sum:
            data = data.groupby(TimeGrouper('D')).sum()
        return jsonify(json.loads(data.to_json(double_precision=round_prec)))
    return abort(500)


#############################
# ENERPI SERVER COMMAND
#############################
@app.route('/api/restart/<service>', methods=['POST'])
@auto.doc()
def startstop(service='enerpi_start'):
    """
    Endpoint for control ENERPI in RPI. Only for dev mode.
    It can restart the ENERPI daemon logger or even reboot the machine for a fresh start after a config change.

    :param service: service id ('enerpi_start/stop' for operate with the logger, or 'machine' for a reboot)

    """
    def _system_operation(command):
        log('SYSTEM_OPERATION CMD: "{}"'.format(command), 'debug', False)
        os.system(command)

    form = DummyForm()
    cmd = msg = alert = None
    if form.validate_on_submit():
        if service == 'enerpi_start':
            python_pathbin = os.path.dirname(sys.executable)
            cmd = '{}/enerpi-daemon start'.format(python_pathbin)
            msg = 'Starting ENERPI logger from webserver... ({})'.format(cmd)
            alert = 'warning'
        elif service == 'enerpi_stop':
            python_pathbin = os.path.dirname(sys.executable)
            cmd = '{}/enerpi-daemon stop'.format(python_pathbin)
            msg = 'Stopping ENERPI logger from webserver... ({})'.format(cmd)
            alert = 'danger'
        elif service == 'machine':
            cmd = 'reboot now'
            msg = 'Rebooting! MACHINE... see you soon... ({})'.format(cmd)
            alert = 'danger'
        if cmd is not None:
            log(msg, 'debug', False)
            t = Timer(.5, _system_operation, args=(cmd,))
            t.start()
            return redirect(url_for('control', after_sysop=True,
                                    alerta=json.dumps({'alert_type': alert, 'texto_alerta': msg})))
    return abort(500)
