# -*- coding: utf-8 -*-
"""
Tests ENERPI pushbullet notifications

"""
from unittest import TestCase


class TestEnerpiNotifier(TestCase):

    def test_pushbullet_push_text(self):
        from enerpi.base import SENSORS
        from enerpi.notifier import push_enerpi_error, push_enerpi_channel_msg

        push_enerpi_channel_msg('Testing channel', repr(SENSORS))
        push_enerpi_error('Testing error', 'Error message\nwith multi-line\ns\nu\np\np\no\nr\nt\n')
