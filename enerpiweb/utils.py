# -*- coding: utf-8 -*-
"""
Flask filters for jinja2 rendering and flask error handlers

"""
import datetime as dt
from flask import request, render_template
from enerpi.base import log
from enerpiweb import app


@app.template_filter('color')
def filter_color(color):
    """
    jinja2 template filter for color conversion to CSS text
    :param color: color as 3/4 items tuple/list, or as hex color like '#aabbcc' or 'aabbcc'
    :return: :str: css color

    """
    if (type(color) is list) or (type(color) is tuple):
        rgb = [str(int(x)) for x in color[:3]]
        if len(color) == 3:
            return 'rgb({})'.format(', '.join(rgb))
        else:
            return 'rgba({})'.format(', '.join(rgb + [str(color[3])]))
    else:
        assert type(color) is str
        if not color.startswith('#'):
            return '#{}'.format(color)
        return color


@app.template_filter('text_date')
def text_date(str_date):
    """
    jinja2 template filter for date conversion from:
        # of days --> today +- #days
        'today' / 'yesterday' --> proper conversion

    :param str_date: :str: datestr
    :return: :str: datestr

    """
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
    """
    strftime jinja2 template filter

    :param ts: :datetime: datetime object
    :return: :str: datestr

    """
    try:
        if (ts.hour == 0) and (ts.minute == 0):
            return ts.strftime('%d/%m/%y')
        return ts.strftime('%d/%m/%y %H:%M')
    except AttributeError as e:
        log('AttributeError en template_filter:ts_strftime -> {}'.format(e), 'error', False)
        return str(ts)


@app.template_filter('docstrip')
def docstrip(str_doc):
    """
    multi-line text stripper (jinja2 template filter)

    :param str_doc:
    :return: tunned str

    """
    # print('DEBUG en docstrip: ', str_doc)
    str_doc = str_doc.rstrip().rstrip('\n')
    str_doc = str_doc.lstrip().lstrip('\n')
    str_doc = str_doc.replace('\n', '<br>')
    # print('DEBUG docstrip RESULT: ', str_doc)
    return str_doc


# @app.errorhandler(InvalidAPIUsage)
# def handle_invalid_usage(error):
#     response = jsonify(error.to_dict())
#     response.status_code = error.status_code
#     return response


@app.errorhandler(404)
def page_not_found(error):
    """
    Error 404
    :param error: error

    """
    log('page_not_found: {}, url={}'.format(error, request.url), 'error', False)
    return render_template('error.html', error_e=error, code=404, traceback=None), 404


@app.errorhandler(500)
def internal_server_error(e):
    """
    Error 500
    :param e: error

    """
    log('INTERNAL_SERVER_ERROR: {}, request={}'.format(e, request), 'error')
    return render_template('error.html', error_e=e, code=500, traceback=None), 500
