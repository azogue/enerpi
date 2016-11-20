# -*- coding: utf-8 -*-
from enerpi.ledrgb import *
from time import sleep


def test_ledrgb():
    # PINOUT RGB_LED
    # WITH_RGBLED = CONFIG.getboolean('RGBLED', 'WITH_RGBLED', fallback=False)
    # PIN_R = CONFIG.getint('RGBLED', 'PIN_R', fallback=19)
    # PIN_G = CONFIG.getint('RGBLED', 'PIN_G', fallback=20)
    # PIN_B = CONFIG.getint('RGBLED', 'PIN_B', fallback=16)

    led = get_rgbled(verbose=True)
    if led is None:
        print('ERROR!! NO RGB LED. (WITH_LED={}; PINOUT_RGB=({},{},{})'.format(WITH_RGBLED, PIN_R, PIN_G, PIN_B))
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


if __name__ == '__main__':
    test_ledrgb()
