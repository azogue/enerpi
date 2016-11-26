# -*- coding: utf-8 -*-
from flask import request, redirect, url_for, render_template, jsonify, abort
import json
from threading import Timer
import os
import sys
from enerpi.base import log
from enerpi.api import enerpi_data_catalog
from enerpiplot.plotbokeh import get_bokeh_version
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
    is_sender_active, last = stream_is_alive()
    after_sysop = request.args.get('after_sysop', '')
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
    form_operate = DummyForm()
    return render_template('control_panel.html',
                           d_catalog={'path_raw_store': os.path.join(cat.base_path, cat.raw_store),
                                      'path_catalog': os.path.join(cat.base_path, cat.catalog_file),
                                      'ts_init': cat.min_ts, 'ts_catalog': cat.index_ts},
                           d_last_msg=last, is_sender_active=is_sender_active, list_stores=paths_rel,
                           form_operate=form_operate, after_sysop=after_sysop,
                           alerta=alerta)


@app.route('/api/help', methods=['GET'])
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
    return auto.html(template='doc/api_help.html', title='enerPI Help')


@app.route('/api/bokehplot', methods=['GET'])
@auto.doc()
def bokehplot():
    """
    Base webpage for query & show bokeh plots of ENERPI data

    """
    return render_template('bokeh_plot.html', url_stream_bokeh=url_for('bokeh_buffer'), b_version=BOKEH_VERSION)


@app.route('/', methods=['GET'])
def base_index():
    """
    Redirects to 'index', with real-time monitoring tiles of ENERPI sensors

    """
    return redirect(url_for('index'))


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
