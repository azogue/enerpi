# -*- coding: utf-8 -*-
"""
Flask routes for send emails with Flask-Mail

    * GMAIL ACCOUNT:        enerpi.bot@gmail.com (hardcoded in enerpi.base with pw)

"""
from flask import redirect, url_for, render_template
from flask_mail import Message
import json
import os
from enerpi.base import log, DATA_PATH, RECIPIENT, SENSORS, INDEX_DATA_CATALOG, async_task
from enerpiweb import app, auto, mail, STATIC_PATH, GMAIL_ACCOUNT


#############################
# ENERPI MAILING
#############################
@async_task
def _send_status_email(text_msg, recipients):
    log('SENDING STATUS EMAIL TO {}'.format(recipients), 'debug')
    mask = os.path.join(STATIC_PATH, "img/generated/tile_enerpi_data_{}_last_24h.svg")
    data_monitor = SENSORS.to_dict()
    for i, k in enumerate(data_monitor['sensors']):
        try:
            data_monitor['sensors'][i]['tile'] = open(mask.format(data_monitor['sensors'][i]['name']), 'r').read()
        except OSError:
            data_monitor['sensors'][i]['tile'] = 'NO TILE'
    try:
        data_monitor['consumption'] = {'tile': open(mask.format('kWh'), 'r').read()}
    except OSError:
        data_monitor['consumption'] = {'tile': 'NO kWh TILE'}
    try:
        data_monitor['ref'] = {'tile': open(mask.format('ref'), 'r').read()}
    except OSError:
        data_monitor['ref'] = {'tile': 'NO REF TILE'}
    html_msg = render_template('email/status_email.html', data_monitor=data_monitor, msg=text_msg)

    msg = Message("Status Info",
                  recipients=recipients,
                  sender='"enerPI" <{}>'.format(GMAIL_ACCOUNT),
                  # cc=None,
                  # bcc=GMAIL_ACCOUNT,
                  # attachments=None,
                  reply_to='"enerPI" <{}>'.format(GMAIL_ACCOUNT),
                  # date=None,
                  body=text_msg, html=html_msg)
    if os.path.exists(INDEX_DATA_CATALOG):
        with open(os.path.join(DATA_PATH, INDEX_DATA_CATALOG), 'rb') as fp:
            msg.attach(INDEX_DATA_CATALOG, "text /csv", fp.read())
    mail.send(msg)
    log('EMAIL SENDED', 'debug')


@app.route('/api/email/status', methods=['GET'])
@app.route('/api/email/status/<recipients>', methods=['GET'])
@auto.doc()
def send_status_email(recipients=(RECIPIENT,)):
    """
    GET method for send the enerPI status from last 24 hours to the specified email recipients.

    :param recipients: comma separated emails, like: '/api/email/status/example@hotmail.com,eugenio.panadero@gmail.com'
    :return: Start the sending email process in async mode, and redirects to '/control' with an alert.

    """
    if type(recipients) is str:
        recipients = recipients.split(',')
    if recipients[0]:
        text_msg = 'enerPI STATUS from last 24h... '
        _send_status_email(text_msg, recipients)
        alert = json.dumps({'alert_type': 'success',
                            'texto_alerta': ('Sended email to: {}\n->"{}"'
                                             .format(recipients if len(recipients) > 1 else recipients[0], text_msg))})
    else:
        alert = json.dumps({'alert_type': 'danger',
                            'texto_alerta': "Can't send email, no default EMAIL RECIPIENT defined"})
    return redirect(url_for('control', alerta=alert))
