# -*- coding: utf-8 -*-
"""
Flask routes for handle with log & config ENERPI files

"""
from flask import request, redirect, url_for, render_template, send_file, abort
import json
import os
from enerpi.base import (get_lines_file, log, DATA_PATH, FILE_LOGGING, SERVER_FILE_LOGGING_RSCGEN,
                         UWSGI_CONFIG_FILE, DAEMON_STDOUT, DAEMON_STDERR)
from enerpi.editconf import web_config_edition_data, check_uploaded_config_file, ENERPI_CONFIG_FILES
from enerpi.api import enerpi_data_catalog
from enerpiweb import app, auto, SERVER_FILE_LOGGING


def _get_filepath_from_file_id(file_id):
    # Interesting files / logs to show/edit/download/upload:
    if file_id in ENERPI_CONFIG_FILES:
        return True, False, os.path.join(DATA_PATH, ENERPI_CONFIG_FILES[file_id]['filename'])
    elif 'flask' == file_id:
        is_logfile, filename = True, SERVER_FILE_LOGGING
    elif 'rsc' == file_id:
        is_logfile, filename = True, SERVER_FILE_LOGGING_RSCGEN
    elif 'nginx_err' == file_id:
        is_logfile, filename = True, '/var/log/nginx/error.log'
    elif 'nginx' == file_id:
        is_logfile, filename = True, '/var/log/nginx/access.log'
    elif 'enerpi' == file_id:
        is_logfile, filename = True, FILE_LOGGING
    elif 'uwsgi' == file_id:
        is_logfile, filename = True, '/var/log/uwsgi/{}.log'.format(os.path.splitext(UWSGI_CONFIG_FILE)[0])
    elif 'daemon_out' == file_id:
        is_logfile, filename = True, DAEMON_STDOUT
    elif 'daemon_err' == file_id:
        is_logfile, filename = True, DAEMON_STDERR
    else:  # Fichero derivado del catálogo
        cat = enerpi_data_catalog(check_integrity=False)
        if 'raw_store' == file_id:
            is_logfile, filename = False, os.path.join(cat.base_path, cat.raw_store)
        elif 'catalog' == file_id:
            is_logfile, filename = False, os.path.join(cat.base_path, cat.catalog_file)
        else:
            log('FILE_ID No reconocido: {}'.format(file_id), 'error', False)
            is_logfile, filename = False, SERVER_FILE_LOGGING
            return False, is_logfile, filename
    return True, is_logfile, filename


#############################
# ROUTES for file handling
#############################
@app.route('/api/hdfstores/<relpath_store>', methods=['GET'])
@auto.doc()
def download_hdfstore_file(relpath_store=None):
    """
    Download HDFStore file from ENERPI (*.h5 file)

    :param str relpath_store: HDF store filename

    """
    cat = enerpi_data_catalog(check_integrity=False)
    path_file = cat.get_path_hdf_store_binaries(relpath_store)
    log('download_hdfstore_file with path_file: "{}", relpath: "{}"'.format(path_file, relpath_store), 'debug', False)
    if (path_file is not None) and os.path.exists(path_file):
        if 'as_attachment' in request.args:
            return send_file(path_file, as_attachment=True, attachment_filename=os.path.basename(path_file))
        return send_file(path_file, as_attachment=False)
    return abort(404)


@app.route('/api/filedownload/<file_id>', methods=['GET'])
@auto.doc()
def download_file(file_id):
    """
    File Download for identified log or config files

    :param str file_id:

    """
    ok, _, filename = _get_filepath_from_file_id(file_id)
    if ok:
        if os.path.exists(filename):
            if 'as_attachment' in request.args:
                return send_file(filename, as_attachment=True, attachment_filename=os.path.basename(filename))
            return send_file(filename, as_attachment=False)
        else:
            msg = json.dumps({'alert_type': 'danger',
                              'texto_alerta': 'El archivo "{}" ({}) no existe!'.format(filename, file_id)})
            return redirect(url_for('control', alerta=msg))
    return abort(404)


@app.route('/api/showfile', methods=['GET'])
@app.route('/api/showfile/<file>', methods=['GET'])
@auto.doc()
def showfile(file='flask'):
    """
    Página de vista de fichero de texto, con orden ascendente / descendente y/o nº de últimas líneas ('tail' de archivo)

    :param str file: file_id to show

    """
    ok, is_logfile, filename = _get_filepath_from_file_id(file)
    if ok:
        delete = request.args.get('delete', '')
        reverse = request.args.get('reverse', False)
        tail_lines = request.args.get('tail', None)
        alerta = request.args.get('alerta', '')
        if alerta:
            alerta = json.loads(alerta)
        if not alerta and delete and is_logfile:
            with open(filename, 'w') as f:
                f.close()
            cad_delete = 'LOGFILE {} DELETED'.format(filename.upper())
            log(cad_delete, 'warn', False)
            return redirect(url_for('showfile', file=file,
                                    alerta=json.dumps({'alert_type': 'warning', 'texto_alerta': cad_delete})))
        data = get_lines_file(filename, tail=tail_lines, reverse=reverse)
        return render_template('text_file.html', titulo='LOG File:' if is_logfile else 'CONFIG File:', file_id=file,
                               subtitulo='<strong>{}</strong>'.format(filename), is_logfile=is_logfile,
                               file_content=data, filename=os.path.basename(filename), alerta=alerta)
    return abort(404)


@app.route('/api/editconfig/', methods=['GET'])
@app.route('/api/editconfig/<file>', methods=['GET', 'POST'])
@auto.doc()
def editfile(file='config'):
    """
    Configuration editor, for INI file, JSON sensors file & encripting key

    :param str file: file_id to edit

    """
    ok, is_logfile, filename = _get_filepath_from_file_id(file)
    if ok and (file in ENERPI_CONFIG_FILES):
        alerta = request.args.get('alerta', '')
        if alerta:
            alerta = json.loads(alerta)
        extra_links = [(ENERPI_CONFIG_FILES[f_id]['text_button'], url_for('editfile', file=f_id))
                       for f_id in sorted(ENERPI_CONFIG_FILES.keys(), key=lambda x: ENERPI_CONFIG_FILES[x]['order'])]
        show_switch_comments = ENERPI_CONFIG_FILES[file]['show_switch_comments']
        if not show_switch_comments:
            without_comments = True
        else:
            without_comments = request.args.get('without_comments', 'False').lower() == 'true'
        d_edit = request.form if request.method == 'POST' else None
        alerta_edit, lines_config, config_data = web_config_edition_data(file, filename, d_edition_form=d_edit)
        if not alerta:
            alerta = alerta_edit
        elif alerta_edit is not None:
            alerta.update(alerta_edit)
        return render_template('edit_text_file.html', titulo='CONFIG EDITOR', file_id=file,
                               show_switch_comments=show_switch_comments, with_comments=not without_comments,
                               abspath=filename, dict_config_content=config_data, file_lines=lines_config,
                               filename=os.path.basename(filename), alerta=alerta, extra_links=extra_links)
    log('Error in editfile with file={}'.format(file), 'error', False)
    return abort(404)


@app.route('/api/uploadfile/<file>', methods=['POST'])
@auto.doc()
def uploadfile(file):
    """
    POST method for interesting config files upload & replacement

    :param str file: uploaded file_id

    """
    ok_fileid, is_logfile, filename = _get_filepath_from_file_id(file)
    if ok_fileid:
        if is_logfile:
            log('uploading logfile {} [{}]!!!'.format(file, filename), 'error', True)
            return redirect(url_for('index'), code=404)
        f = request.files['file']
        alert = check_uploaded_config_file(file, f, dest_filepath=filename)
        if alert:
            alert = json.dumps(alert)
        return redirect(url_for('editfile', file=file, alerta=alert))
    return abort(500)
