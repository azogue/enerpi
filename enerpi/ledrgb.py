# -*- coding: utf-8 -*-
from enerpi.base import CONFIG, log, CONFIG_FILENAME


# PINOUT RGB_LED
WITH_RGBLED = CONFIG.getboolean('RGBLED', 'WITH_RGBLED', fallback=False)
PIN_R = CONFIG.getint('RGBLED', 'PIN_R', fallback=19)
PIN_G = CONFIG.getint('RGBLED', 'PIN_G', fallback=20)
PIN_B = CONFIG.getint('RGBLED', 'PIN_B', fallback=16)


def get_rgbled(verbose=True):
    """Tries to get gpiozero RGBLED object at pins PIN_R|G|B if WITH_RGBLED in config file"""
    led = None
    if WITH_RGBLED:
        try:
            from gpiozero import RGBLED

            led = RGBLED(PIN_R, PIN_G, PIN_B, active_high=True)
            led.blink(.25, .25, on_color=(1, 1, 1), n=5)
        except (OSError, RuntimeError, ImportError) as e:
            log('** Not using RGBLED with GPIOZERO ({} [{}]). Check your "{}" file.'
                .format(e, e.__class__, CONFIG_FILENAME), 'warn', verbose)
    return led


def led_alarm(led, time_blinking=2.5, timeout=3):
    """Blinks in RED (one time or multiple times, always with ∆T=.25 sec)"""
    if timeout == 0:
        led.blink(.25, .25, on_color=(1, 0, 0))
    else:
        led.blink(.25, .25, on_color=(1, 0, 0), n=int(time_blinking / .5))


def led_info(led, n=3):
    """Blinks in BLUE n times with ∆T=.5 sec"""
    led.blink(.5, .5, on_color=(0, 0, 1), n=n)


def blink_color(led, color, n=1):
    """Blinks n times in color with ∆T=.5 sec"""
    led.blink(.5, .5, on_color=color, n=n)


