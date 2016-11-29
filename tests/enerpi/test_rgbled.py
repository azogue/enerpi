# -*- coding: utf-8 -*-
"""
Test RGBLED

"""
from time import sleep
from tests.conftest import TestCaseEnerpi


class TestEnerpiLEDRGB(TestCaseEnerpi):

    def test_ledrgb(self):
        """
        RGB LED Testing
        :return:
        """
        from enerpi.base import CONFIG
        from enerpi.ledrgb import get_rgbled, led_alarm, led_info, blink_color

        led = get_rgbled(verbose=True)
        if led is None:
            # PINOUT RGB_LED
            with_rgbled = CONFIG.getboolean('RGBLED', 'WITH_RGBLED', fallback=False)
            pin_r = CONFIG.getint('RGBLED', 'PIN_R', fallback=19)
            pin_g = CONFIG.getint('RGBLED', 'PIN_G', fallback=20)
            pin_b = CONFIG.getint('RGBLED', 'PIN_B', fallback=16)
            print('ERROR!! NO RGB LED. (WITH_LED={}; PINOUT_RGB=({},{},{})'.format(with_rgbled, pin_r, pin_g, pin_b))
            # assert 0
        else:
            print('TESTING RGB LED!')
            led_alarm(led, timeout=0)
            sleep(1)
            led_info(led, n=3)
            sleep(3)
            led_alarm(led, time_blinking=2.5, timeout=3)
            sleep(4)
            led_info(led, n=2)
            sleep(4)
            led_alarm(led, timeout=0)
            sleep(2)
            led_info(led, n=1)
            sleep(3)

            # Blinking:
            blink_color(led, (1, 0, 0), n=1)
            sleep(1)
            blink_color(led, (0, 1, 0), n=2)
            sleep(1)
            blink_color(led, (0, 0, 1), n=3)
            sleep(1)
            blink_color(led, (1, 1, 0), n=1)
            sleep(1)
            blink_color(led, (1, 0, 1), n=2)
            sleep(1)
            blink_color(led, (0, 1, 1), n=3)
            sleep(1)
            blink_color(led, (1, 1, 1), n=1)
            sleep(2)
            print('OK!!')
            led.close()


if __name__ == '__main__':
    import unittest

    unittest.main()
