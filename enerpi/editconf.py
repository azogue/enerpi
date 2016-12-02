# -*- coding: utf-8 -*-
"""
Methods for edit configuration files in ENERPI WebServer, with some validation

    - web_edit_* := from filepath to lines_file + dict_form_template
    - web_post_* := from posted form with changes and original config for saving user config

"""
from collections import OrderedDict
from io import BytesIO
import json
from jsondiff import diff
import os
import textwrap
from enerpi.base import log, get_lines_file, ENCODING, DATA_PATH, CONFIG
from enerpi.iobroadcast import get_codec


ENERPI_CONFIG_FILES = {
    'config': {
        'order': 0,
        'filename': 'config_enerpi.ini',
        'text_button': '<i class="fa fa-check-square-o" aria-hidden="true"></i> ENERPI config',
        'show_switch_comments': True,
    },
    'sensors': {
        'order': 1,
        'filename': 'sensors_enerpi.json',
        'text_button': '<i class="fa fa-wrench" aria-hidden="true"></i> SENSORS',
        'show_switch_comments': False,
    },
    'encryption_key': {
        'order': 2,
        'filename': os.path.join(DATA_PATH, CONFIG.get('BROADCAST', 'KEY_FILE', fallback='.secret_key')),
        'text_button': '<i class="fa fa-user-secret" aria-hidden="true"></i> Secret Key',
        'show_switch_comments': False,
    },
}

TITLE_EDIT_CRYPTOKEY = 'Encryption KEY'
SUBTITLE_EDIT_CRYPTOKEY = 'KEY_FILE'
TITLE_EDIT_JS_SENSORS = 'SENSORS'
SUBTITLE_EDIT_JS_SENSORS = 'SENSORS'

OPTIONAL_PARAMS = ['PUSHBULLET_TOKEN', 'EMAIL_RECIPIENT']


def _web_edit_enerpi_config_ini(lines_ini_file):
    """
    * Divide configuration INI file with this structure:
    [section]
    # comment
    ; comment
    VARIABLE = VALUE
    (discarding file header comments)

    Make Ordered dict like:
    ==> [(section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         (section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         ...]

    :param lines_ini_file: Text lines of INI file
    :return: :tuple of (lines_file_for_webview, dict_file_for_webform):

    """
    def _is_bool_variable(value):
        value = value.lower()
        if (value == 'true') or (value == 'false'):
            return True, value == 'true'
        return False, None

    config_entries = OrderedDict()
    section, comment = None, None
    init = False
    for l in lines_ini_file:
        l = l.replace('\n', '').lstrip().rstrip()
        if l.startswith('['):
            section = l.replace('[', '').replace(']', '')
            init = True
            comment = None
            config_entries[section] = OrderedDict()
        elif l.startswith('#') or l.startswith(';'):
            if init:
                if comment is None:
                    comment = l[1:].lstrip()
                else:
                    comment += ' {}'.format(l[1:].lstrip())
        elif init and (len(l) > 0):
            # Read variable and append (/w comments)
            variable_name, variable_value = l.split('=')
            variable_name = variable_name.lstrip().rstrip()
            variable_value = variable_value.lstrip().rstrip()
            is_bool, bool_value = _is_bool_variable(variable_value)
            if not is_bool:
                try:
                    variable_value = int(variable_value)
                    var_type = 'int'
                except ValueError:
                    try:
                        variable_value = float(variable_value)
                        var_type = 'float'
                    except ValueError:
                        var_type = 'str'
            else:
                var_type = 'bool'
                variable_value = bool_value
            config_entries[section][variable_name] = variable_value, var_type, comment
            comment = None
    return True, config_entries


def _web_edit_enerpi_sensors_json(lines_json):
    """
    * Get JSON sensors config for web user edition:

    Make Ordered dict like:
    ==> [(section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         (section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         ...]

    :param lines_json: Lines of JSON file
    :return: :tuple of (ok, dict_file_for_webform):

    """
    # TODO JSON EDITOR
    try:
        sensors_json = json.loads('\n'.join(lines_json), encoding=ENCODING)
        t, sub = TITLE_EDIT_JS_SENSORS, SUBTITLE_EDIT_JS_SENSORS
        d_conf = OrderedDict([(t, OrderedDict([(sub, (json.dumps(sensors_json, indent=1), 'text', None))]))])
        return True, d_conf
    except json.decoder.JSONDecodeError as e:
        msg = 'JSONDecodeError reading new SENSORS JSON: {} --> {}'.format(e, lines_json)
        log(msg, 'error')
        return False, {'error': msg}


def _web_edit_enerpi_encryption_key(key_file_lines):
    """
    Return dict for user edition of encryption key
    Make Ordered dict like:
    ==> [(section_name,
            OrderedDict([(VARIABLE, (VALUE, 'text', comment=None))])]

    :param key_file_lines: Text lines of encryption key file (1)
    :return: :tuple of (lines_file_for_webview, dict_file_for_webform):

    """
    try:
        assert len(key_file_lines) == 1
        key = key_file_lines[0]
    except AssertionError:
        msg = 'ERROR Reading CRYPTO Key (incorrect # of lines): {}'.format(key_file_lines)
        log(msg, 'error', False)
        return False, {'error': msg}
    try:
        _ = get_codec(key.encode())
    except AssertionError as e:
        msg = 'ASSERT WITH CRYPTO_KEY: {}, KEY="{}"'.format(e, key)
        log(msg, 'error', False)
        return False, {'error': msg}
    t, sub = TITLE_EDIT_CRYPTOKEY, SUBTITLE_EDIT_CRYPTOKEY
    return [key], OrderedDict([(t, OrderedDict([(sub, (key, 'text', None))]))])


def _make_ini_file(dict_config, dest_path=None):
    """
    Makes INI file (and writes it if dest_path is not None) from an OrderedDict
    like the one 'config_dict_for_web_edit' returns.

    * INI file with this structure:
    [section]
    # comment
    ; comment
    VARIABLE = VALUE

    Ordered dict like:
    ==> [(section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         (section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         ...]

    :param dict_config: :OrderedDict: Content of INI file after 'readlines()'
    :param dest_path: :str: optional path for INI file write.
    :return: :str: raw INI text

    """
    lines = ['# -*- coding: utf-8 -*-']
    for section, entries in dict_config.items():
        lines.append('[{}]'.format(section.upper()))
        for variable_name, (value, var_type, comment) in entries.items():
            if comment:
                [lines.append('# {}'.format(l_wrap)) for l_wrap in textwrap.wrap(comment, 80)]
            lines.append('{} = {}'.format(variable_name, value))
        lines.append('')  # Separador entre secciones
    ini_text = '\n'.join(lines)
    if dest_path is not None:
        with open(dest_path, 'w') as f:
            f.write(ini_text)
    return ini_text


def _web_post_changes_enerpi_config_ini(dict_web_form, lines_config, dict_config, dest_filepath=None):
    """
    Process changes in web editor of INI ENERPI config file

    :param dict_web_form: :OrderedDict: Posted Form values
    :param lines_config:  :list: original file text lines
    :param dict_config:  :OrderedDict: original config dict of dicts (like the one 'web_edit_enerpi_config_ini' returns)
    :param dest_filepath:  :str: (optional) destination filepath for save configuration changes
    :return: :tuple: (:dict: alert message, :list: text lines, :OrderedDict: updated dict_config))

    """
    def _is_changed(value, params, name):
        if (value is None) or ((len(value) == 0) and (name not in OPTIONAL_PARAMS)):
            msg = 'INI Value ERROR: key={}, value="{}", (type={})'.format(name, value, params[1])
            return False, value, msg
        try:
            if params[1] == 'int':
                try:
                    value = int(value)
                except ValueError:
                    value = float(value)
            elif params[1] == 'float':
                value = float(value)
            elif params[1] == 'bool':
                value = (value.lower() == 'true') or (value.lower() == 'on')
            if value != params[0]:
                log('"{}" -> HAY CAMBIO DE {} a {} (type={})'.format(name, params[0], value, params[1]), 'debug', False)
                return True, value, None
            return False, value, None
        except ValueError as e:
            msg = 'ValueError reading INI: key={}, value={}, (type={}). Error: {}'.format(name, value, params[1], e)
            return False, value, msg

    dict_config_updated = dict_config.copy()
    dict_web_form = dict_web_form.copy()
    vars_updated = []
    for section, entries in dict_config_updated.items():
        for variable_name, variable_params in entries.items():
            if variable_name in dict_web_form:
                new_v = dict_web_form.pop(variable_name)
                changed, new_value, error = _is_changed(new_v, variable_params, variable_name)
                if error is not None:
                    alerta = {'alert_type': 'danger', 'texto_alerta': error}
                    return alerta, lines_config, dict_config
                if changed:
                    vars_updated.append((variable_name, variable_params[0], new_value))
                    params_var = list(dict_config_updated[section][variable_name])
                    params_var[0] = new_value
                    dict_config_updated[section][variable_name] = tuple(params_var)
            elif (variable_params[1] == 'bool') and variable_params[0]:  # Bool en off en el form y True en config
                vars_updated.append((variable_name, variable_params[0], False))
                params_var = list(dict_config_updated[section][variable_name])
                params_var[0] = False
                log('"{}" -> HAY CHECKBOX CH DE {} a {} (type={})'
                    .format(variable_name, variable_params[0], False, variable_params[1]), 'debug', False)
                dict_config_updated[section][variable_name] = tuple(params_var)
    alerta = None
    if len(vars_updated) > 0:
        str_cambios = ('Configuration changes in:<br>{}<br> New config SAVED!'
                       .format('<br>'.join(['- <strong>"{}"</strong> (before={}) -> <strong>"{}"</strong>'
                                           .format(name, ant, new) for name, ant, new in vars_updated])))
        alerta = {'alert_type': 'warning', 'texto_alerta': str_cambios}
        log(str_cambios, 'debug', False)
        lines_config = _make_ini_file(dict_config_updated, dest_path=dest_filepath).splitlines()
    return alerta, lines_config, dict_config_updated


def _web_post_changes_enerpi_sensors_json(dict_web_form, lines_config, dict_config, dest_filepath=None):
    """
    Process changes in web editor of ENERPI SENSORS JSON File

    :param dict_web_form: :OrderedDict: Posted Form with new JSON data
    :param lines_config: :list: original file text lines (1) with original JSON
    :param dict_config: :OrderedDict: original config dict of dicts
                            (like the one 'web_edit_enerpi_sensors_json' returns)
    :param dest_filepath: :str: (optional) destination filepath for save configuration changes
    :return: :tuple: (:dict: alert message, :list: text lines, :OrderedDict: updated dict_config))

    """
    alerta = None
    t, sub = TITLE_EDIT_JS_SENSORS, SUBTITLE_EDIT_JS_SENSORS
    try:
        ant_json = json.loads(dict_config[t][sub][0], encoding=ENCODING)
        new_json = json.loads(dict_web_form[sub], encoding=ENCODING)
        diff_json = diff(ant_json, new_json)
        if diff_json:
            if dest_filepath is not None:
                log('New JSON Sensors config ("{}")\nwill be saved in "{}"'
                    .format(new_json, dest_filepath), 'warning', False)
                with open(dest_filepath, 'w') as f:
                    json.dump(new_json, f, indent=1)
            str_cambios = ('Configuration changes in ENERPI SENSORS:<br>{}<br> New config SAVED!'
                           .format('JSON DIFF- <strong>"{}"</strong>'.format(diff_json)))
            log(str_cambios, 'debug', False)
            alerta = {'alert_type': 'warning', 'texto_alerta': str_cambios}
            if dest_filepath is not None:
                lines_config = get_lines_file(dest_filepath)
                ok, dict_config = _web_edit_enerpi_sensors_json(lines_config)
                if not ok:
                    alerta['alert_type'] = 'error'
                    alerta['texto_alerta'] += dict_config['error']
                    alerta['texto_alerta'] += '\nNEW JSON SENSORS FILE NOT VALID!! FIX IT, PLEASE'
    except json.decoder.JSONDecodeError:
        msg_err = ('JSONDecodeError in web_post_changes_enerpi_sensors_json: {}'.format(dict_config[t][sub]))
        alerta = {'alert_type': 'error', 'texto_alerta': msg_err}
    return alerta, lines_config, dict_config


def _web_post_changes_enerpi_encryption_key(dict_web_form, lines_keyfile, dict_key_file, dest_filepath=None):
    """
    Process changes in web editor of ENERPI BROADCAST ENCRYPTION KEY
    # Secret Key edition!! NOT SAFE AT ALL!! (Debugging purposes)

    :param dict_web_form: :OrderedDict: Posted Form with new crypto key
    :param lines_keyfile: :list: original file text lines (1) with original cryto key
    :param dict_key_file: :OrderedDict: original config dict of dicts
                            (like the one 'web_edit_enerpi_encryption_key' returns)
    :param dest_filepath: :str: (optional) destination filepath for save configuration changes
    :return: :tuple: (:dict: alert message, :list: text lines, :OrderedDict: updated dict_config))

    """
    t, sub = TITLE_EDIT_CRYPTOKEY, SUBTITLE_EDIT_CRYPTOKEY
    ant_key = dict_key_file[t][sub][0]
    new_key = dict_web_form[sub]
    alerta = {}
    if new_key != ant_key:
        try:  # Validate new Fernet Key
            _ = get_codec(new_key.encode())
            if len(new_key) > 10:
                if dest_filepath is not None:
                    log('The new KEY ("{}")\nwill be saved in "{}"\n'.format(new_key, dest_filepath), 'warning', False)
                    with open(dest_filepath, 'wb') as f:
                        f.write(new_key.encode())
                str_cambios = ('Configuration changes in encryption key:<br>{}<br> New config SAVED!'
                               .format('- <strong>"{}"</strong> (before={}) -> <strong>"{}"</strong>'
                                       .format(SUBTITLE_EDIT_CRYPTOKEY, ant_key, new_key)))
                log(str_cambios, 'debug', False)
                alerta = {'alert_type': 'warning', 'texto_alerta': str_cambios}
                lines_keyfile = [new_key]
                dict_key_file = OrderedDict([(t, OrderedDict([(sub, (new_key, 'text', None))]))])
        except AssertionError as e:
            alerta = {'alert_type': 'danger', 'texto_alerta': 'Not a valid KEY: {}. New KEY was: {}'.format(e, new_key)}
    return alerta, lines_keyfile, dict_key_file


def check_uploaded_config_file(file_id, f_obj, dest_filepath=None):
    """Check uploaded configuration file and save it if correct. Return dict for web alert

    :param file_id: config file id
    :param f_obj: :werkzeug.datastructures.FileStorage: uploaded file object
    :param dest_filepath: optional destination path for uploaded file

    :return: (web alert)
    """
    file_name, file_extension = os.path.splitext(f_obj.filename)
    fmem = BytesIO()
    f_obj.save(fmem)
    fmem.seek(0)
    content = fmem.read().decode()
    lines = content.splitlines()
    if len(content) < 20:
        msg = 'Incorrect uploaded file (too short!): {}'.format(content)
        return {'alert_type': 'danger', 'texto_alerta': msg}
    if (file_id == 'config') and (file_extension == '.ini') and (f_obj.mimetype == 'application/octet-stream'):
        ok_file, error = _web_edit_enerpi_config_ini(lines)
    elif (file_id == 'sensors') and (file_extension == '.json') and (f_obj.mimetype == 'application/json'):
        ok_file, error = _web_edit_enerpi_sensors_json(lines)
    elif (file_id == 'encryption_key') and (f_obj.mimetype == 'application/octet-stream'):
        ok_file, error = _web_edit_enerpi_encryption_key(lines)
    else:
        return {'alert_type': 'danger', 'texto_alerta': 'FileId not recognized: {}'.format(file_id)}
    if not ok_file:
        return {'alert_type': 'danger', 'texto_alerta': error['error']}
    if dest_filepath is not None:
        with open(dest_filepath, 'w') as f:
            f.write(content)
    return {'alert_type': 'success',
            'texto_alerta': 'Uploaded file "{}" as replacement in "{}"!'.format(f_obj.filename, dest_filepath)}


def web_config_edition_data(file_id, filepath, d_edition_form=None):
    """
    Method for validate and save changes on configuration files.
    It returns data for jinja2 templates:
        alert message, Data Get JSON sensors config for web user edition:

    :param file_id: Id of config file
    :param filepath: filepath for read/save configuration changes
    :param d_edition_form: request.form dict for config POST requests as:
        [(section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         (section_name,
            OrderedDict([(VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment)),
                         ...
                         (VARIABLE, (VALUE, 'int|float|bool|text', comment))])),
         ...]
    :return: (:dict: alert message, :list: text lines, :OrderedDict: dict_config for jinja2 templates))

    """
    if file_id == 'config':
        f_edit_file = _web_edit_enerpi_config_ini
        f_post_file = _web_post_changes_enerpi_config_ini
    elif file_id == 'sensors':
        f_edit_file = _web_edit_enerpi_sensors_json
        f_post_file = _web_post_changes_enerpi_sensors_json
    else:
        assert file_id == 'encryption_key'
        f_edit_file = _web_edit_enerpi_encryption_key
        f_post_file = _web_post_changes_enerpi_encryption_key
    alerta = None
    lines_config = get_lines_file(filepath)
    ok_lines, config_data = f_edit_file(lines_config)
    if not ok_lines:
        alerta = {'alert_type': 'danger', 'texto_alerta': config_data['error']}
        log('NOT OK in web_config_edition_data: {}'.format(web_config_edition_data), 'warning', False)
    elif d_edition_form is not None:
        alerta, lines_config, config_data = f_post_file(d_edition_form, lines_config, config_data, filepath)
    return alerta, lines_config, config_data
