# -*- coding: utf-8 -*-
"""
Flask routes para facturación del consumo eléctrico en España.

"""
import datetime as dt
from flask import Response, render_template, url_for, request
import json
from time import time

from esiosdata.facturapvpc import FacturaElec, DATOS_ZONAS_IMPUESTOS, DATOS_TIPO_PEAJE
from enerpi.base import log, SENSORS, BILLING_DATA
from enerpi.api import enerpi_data_catalog
from enerpiweb import app, auto


def _format_event_stream(d_msg):
    return 'data: {}\n\n'.format(json.dumps(d_msg))


def _gen_stream_data_factura(start=None, end=None, **kwargs_factura):
    tic = time()
    if start is None:
        # Inicio de mes actual hasta instante actual
        start = dt.datetime.now(tz=SENSORS.TZ).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = dt.datetime.now(tz=SENSORS.TZ)
    cat = enerpi_data_catalog(check_integrity=False)
    df = cat.get_summary(start=start, end=end)
    toc_df = time()
    if (df is not None) and not df.empty and ('kWh' in df):
        consumption = df['kWh']
        # Fix timezone (ya en esiosdata)
        # try:
        #     consumption.index = consumption.index.tz_localize(SENSORS.TZ, ambiguous='infer')
        # except AmbiguousTimeError as e:
        #     consumption.index = consumption.index.tz_localize(SENSORS.TZ, ambiguous='NaT')
        #     consumption = consumption.reindex(DatetimeIndex(start=consumption.index[0], end=consumption.index[-1],
        #                                                     freq='1h', tz=SENSORS.TZ)).interpolate()
        #     log('AmbiguousTimeError ({}) en elec_bill. Se reindexa e interpola el índice.'.format(e), 'error')
        factura = FacturaElec(consumo=consumption, **kwargs_factura)
        data_factura = factura.to_dict(include_text_repr=True, include_html_repr=True)
        toc_p = time()
        msg = 'Factura generada en {:.3f} s; datos en {:.3f} s.'.format(toc_p - toc_df, toc_df - tic)
        log(msg, 'debug', False)
        log('stream_data_factura: STREAM BILL from "{}" to "{}"'.format(start, end), 'debug', False)
        yield _format_event_stream(dict(success=True, factura=data_factura,
                                        took=round(toc_p - tic, 3), took_df=round(toc_df - tic, 3)))
    elif df is not None:
        factura = FacturaElec(start, end, **kwargs_factura)
        data_factura = factura.to_dict(include_text_repr=True, include_html_repr=True)
        toc_p = time()
        msg = 'Factura vacía generada en {:.3f} s; datos en {:.3f} s.'.format(toc_p - toc_df, toc_df - tic)
        log(msg, 'debug', False)
        log('stream_data_factura: STREAM EMPTY BILL from "{}" to "{}"'.format(start, end), 'debug', False)
        yield _format_event_stream(dict(success=True, factura=data_factura, error=msg,
                                        took=round(toc_p - tic, 3), took_df=round(toc_df - tic, 3)))
    else:
        msg = 'No data from {} to {}. CATALOG:\n{}'.format(start, end, cat)
        log(msg, 'debug', False)
        log('stream_data_factura: STREAM ERR NO DATA from "{}" to "{}"'.format(start, end), 'debug', False)
        yield _format_event_stream(dict(success=False, error=msg,
                                        took=round(time() - tic, 3), took_df=round(toc_df - tic, 3)))
    log('CLOSING stream_data_factura from "{}" to "{}" with args={}'.format(start, end, kwargs_factura), 'debug', False)
    yield _format_event_stream('CLOSE')


###############################
# BILLING DATA
###############################
@app.route('/api/billing', methods=['GET'])
@app.route('/api/billing/from/<start>', methods=['GET'])
@app.route('/api/billing/from/<start>/to/<end>', methods=['GET'])
@auto.doc()
def billing_data(start=None, end=None):
    """
    Stream the billing data to make a report.
    Used for load/reload report from user queries.

    :param start: :str: start datetime of data report
    :param end: :str: end datetime of data report

    :return: SSE stream response

    """
    kwargs = dict(start=start, end=end, cups=BILLING_DATA['cups'])
    if 'potencia' in request.args:
        kwargs.update(potencia_contratada=float(request.args['potencia']))
    if 'bono_social' in request.args:
        kwargs.update(con_bono_social=request.args['bono_social'].lower() == 'true')
    if 'impuestos' in request.args:
        zonas = list(DATOS_ZONAS_IMPUESTOS)
        kwargs.update(zona_impuestos=zonas[int(request.args['impuestos'])])
    if 'peaje' in request.args:
        peajes = list(DATOS_TIPO_PEAJE)
        kwargs.update(tipo_peaje=peajes[int(request.args['peaje'])])
    log('BILLING_DATA: {}'.format(kwargs), 'debug', False)
    return Response(_gen_stream_data_factura(**kwargs), mimetype='text/event-stream')


#############################
# BILLING
#############################
@app.route('/api/bills', methods=['GET'])
@auto.doc()
def elec_bill():
    """Endpoint for get the electrical bill (SPAIN Only, PVPC model) of a consumption period."""
    # Start with the current month:
    now = dt.datetime.now()
    ts_ini = '{:%Y-%m-%d}'.format(now.replace(day=1).date())
    ts_fin = '{:%Y-%m-%d}'.format(now.date())
    url_factura_init = url_for('billing_data', peaje=BILLING_DATA['peaje'], impuestos=BILLING_DATA['zona_impuestos'],
                               bono_social=BILLING_DATA['con_bono'], potencia=BILLING_DATA['p_contrato'], )
    log('In base page for BILLING_DATA with init bill in {}'.format(url_factura_init), 'debug', False)
    return render_template('elec_bill.html', url_factura_init=url_factura_init, ts_ini=ts_ini, ts_fin=ts_fin,
                           zonas_impuestos=DATOS_ZONAS_IMPUESTOS, tipos_peaje=DATOS_TIPO_PEAJE, **BILLING_DATA)
