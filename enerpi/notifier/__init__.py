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
from configparser import NoOptionError, NoSectionError
import datetime as dt
from pushbullet import Pushbullet
from pushbullet.errors import InvalidKeyError
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
