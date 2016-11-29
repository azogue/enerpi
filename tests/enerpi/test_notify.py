# -*- coding: utf-8 -*-
"""
Tests ENERPI pushbullet notifications

"""
import os
from unittest import TestCase


class TestEnerpiNotifier(TestCase):

    def test_pushbullet_push_text(self):
        from enerpi.base import SENSORS
        from enerpi.notifier import push_enerpi_error, push_enerpi_channel_msg

        push_enerpi_channel_msg('Testing channel', repr(SENSORS))
        push_enerpi_error('Testing error', 'Error message\nwith multi-line\ns\nu\np\np\no\nr\nt\n')

    def test_pushbullet_push_files(self):
        from enerpi.base import STATIC_PATH, SENSORS
        from enerpi.notifier import push_enerpi_file, RECIPIENT, CHANNEL

        path_file = os.path.join(STATIC_PATH, 'img', 'generated',
                                 'tile_enerpi_data_{}_last_24h.svg'.format(SENSORS.main_column))

        push_enerpi_file(path_file, title='Test TILE PNG CHANNEL', email=None, channel_tag=CHANNEL,
                         svg_conversion_format='png', verbose=True)

        push_enerpi_file(path_file, title='Test TILE PNG', email=RECIPIENT,
                         svg_conversion_format='png', verbose=True)

        push_enerpi_file(path_file, title='Test TILE PDF', email=RECIPIENT,
                         svg_conversion_format='pdf', verbose=True)
