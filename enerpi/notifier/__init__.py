# -*- coding: utf-8 -*-
"""
ENERPI PushBullet Notifier

- Push text message (ENERPI errors) or file.
- Push to email recipient (with a pushbullet acount!) or to a pushbullet channel ('enerpi_notifications')

Pushbullet API config:
    * Name:                 ENERPI Current Meter BOT
    * Pushbullet channel:   enerpi_notifications
    * Channel URL:          https://www.pushbullet.com/channel?tag=enerpi_notifications
    * PUSHBULLET_TOKEN:     Token from pushbullet account, it needs to be defined in 'enerpi_config.ini'

"""
import cairosvg
from configparser import NoOptionError, NoSectionError
import datetime as dt
import os
from pushbullet import Pushbullet
from pushbullet.errors import InvalidKeyError
from time import time
from enerpi.base import CONFIG, log, async_task, RECIPIENT


CHANNEL = 'enerpi_notifications'
PBOBJ = None


def _get_pb_obj():
    # Lazy load
    global PBOBJ

    if PBOBJ is None:
        try:
            pushbullet_token = CONFIG.get('NOTIFY', 'PUSHBULLET_TOKEN')
            if not pushbullet_token:
                raise InvalidKeyError
            PBOBJ = Pushbullet(pushbullet_token)
        except (NoOptionError, NoSectionError, InvalidKeyError) as e:
            log('NO Pushbullet config ({} [{}])'.format(e, e.__class__), 'error', False)
            PBOBJ = None
    return PBOBJ


def _send_direct_push_notification(title, content, email=RECIPIENT):
    pb = _get_pb_obj()
    if (pb is not None) and email:
        pb.push_note(title, content, email=email)
        log('PUSHBULLET NOTIFICATION: {} - {}'.format(title, content), 'debug', False)
        return True
    log('NO PUSHBULLET NOTIFICATION (title={}, email={})'.format(title, email),
        'error', True)
    return False


@async_task
def push_enerpi_error(title_err, msg_error):
    """
    Send an error message notification (vía pushbullet API) to the default recipient.
    :param title_err: error title
    :param msg_error: error text content

    """
    return _send_direct_push_notification('ERROR: {}\n(time:{:%H:%M:%S})'.format(title_err, dt.datetime.now()),
                                          msg_error, email=RECIPIENT)


@async_task
def push_enerpi_channel_msg(title, msg, channel_tag=CHANNEL):
    """
    Send a text notification (vía pushbullet API) to the ENERPI pushbullet channel (timestamped)

    :param title:
    :param msg:
    :param channel_tag: ENERPI CHANNEL

    """
    content = '{}\n(time:{:%H:%M:%S})'.format(msg, dt.datetime.now())
    pb = _get_pb_obj()
    if pb is not None:
        try:
            ch = list(filter(lambda x: x.channel_tag == channel_tag, pb.channels))[0]
        except IndexError:
            ch = None
        if ch:
            ch.push_note(title, content)
            log('PUSHBULLET CHANNEL NOTIFICATION: {} - {}'.format(title, content), 'debug', False)
            return True
    log('NO PUSHBULLET CHANNEL NOTIFICATION (title={}, msg={}, channel={})'
        .format(title, msg, channel_tag), 'error', True)
    return False


@async_task
def push_enerpi_file(path_file, title=None, email=RECIPIENT, channel_tag=None,
                     svg_conversion_format='png', verbose=False):
    """
    Send a file with pushbullet (to a recipient or to a channel

    :param path_file:
    :param title:
    :param email:
    :param channel_tag:
    :param svg_conversion_format:
    :param verbose:

    """
    pb = _get_pb_obj()
    if (pb is None) or ((channel_tag is None) and not email):
        log('NO PUSHBULLET FILE PUSH (title={}, file={}, email={}, channel={})'
            .format(title, path_file, email, channel_tag), 'error', True)
        return False
    elif not os.path.exists(path_file):
        log('Trying to push a non existent file! ->'.format(path_file), 'error', verbose)
        return False
    tic = time()
    name_file = os.path.basename(path_file)
    # SVG to PNG if needed:
    if path_file.endswith('.svg') and svg_conversion_format:
        name_file = name_file[:-3] + svg_conversion_format
        log('SVG to {} Conversion: {} --> {}'
            .format(svg_conversion_format.upper(), path_file, name_file), 'warning', verbose)
        with open(path_file, 'r') as f:
            svg_content = f.read()
        # SVG Transformation --> some background
        style = "fill:none;opacity:0;"
        new_style = "fill:#FFFFFF;opacity:0.1;"
        svg_content = svg_content.replace(style, new_style)
        # Cairo conversion
        func_convers = getattr(cairosvg, 'svg2{}'.format(svg_conversion_format.lower()))
        b_content_file = func_convers(bytestring=svg_content.encode(), dpi=300, unsafe=True)
    else:
        with open(path_file, 'rb') as f:
            b_content_file = f.read()
    tic_i = time()
    file_data = pb.upload_file(b_content_file, name_file)
    toc = time()
    log('PushBullet file upload: {} (name_pic= {}). Took {:.3f} sec'
        .format(path_file, name_file, toc - tic), 'info', verbose)
    if title is not None:
        file_data.update({'title': title})
    if email is not None:
        file_data.update({'email': email})
    else:
        try:
            ch = list(filter(lambda x: x.channel_tag == channel_tag, pb.channels))[0]
        except IndexError:
            ch = None
        if ch:
            file_data.update({'channel': ch})
        else:
            log('No se encuentra el canal: "{}". Los disponibles son:\n{}'
                .format(channel_tag, pb.channels), 'error', verbose)
            return False
    # Envía notificación
    pb.push_file(**file_data)
    log('PB_NOTIF TOOK: PROCESS_FILE={:.2f}s; UPLOAD_FILE={:.2f}s; PUSH_FILE={:.2f}s'
        .format(tic_i - tic, toc - tic_i, time() - toc), 'debug', verbose)
    return True
