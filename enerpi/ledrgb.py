# -*- coding: utf-8 -*-
"""
ENERPI - RGB LED control methods

"""
from enerpi.base import CONFIG, log, CONFIG_FILENAME
import datetime as dt
from dateutil.parser import parse


# PINOUT RGB_LED
WITH_RGBLED = CONFIG.getboolean('RGBLED', 'WITH_RGBLED', fallback=False)
PIN_R = CONFIG.getint('RGBLED', 'PIN_R', fallback=19)
PIN_G = CONFIG.getint('RGBLED', 'PIN_G', fallback=20)
PIN_B = CONFIG.getint('RGBLED', 'PIN_B', fallback=16)

# Horario de funcionamiento del RGB LED
LED_TIME_ON = CONFIG.get('RGBLED', 'TIME_ON', fallback=None)
LED_TIME_OFF = CONFIG.get('RGBLED', 'TIME_OFF', fallback=None)
if LED_TIME_ON is not None:
    LED_TIME_ON = parse(LED_TIME_ON)
if LED_TIME_OFF is not None:
    LED_TIME_OFF = parse(LED_TIME_OFF)


def _use_led():
    if LED_TIME_ON and LED_TIME_OFF:
        if LED_TIME_ON.time() < dt.datetime.now().time() < LED_TIME_OFF.time():
            return True
        return False
    return True


def get_rgbled(verbose=True):
    """Tries to get gpiozero RGBLED object at pins PIN_R|G|B if WITH_RGBLED in config file"""
    led = None
    if WITH_RGBLED:
        try:
            from gpiozero import RGBLED
            from gpiozero.pins.rpigpio import RPiGPIOPin

            led = RGBLED(RPiGPIOPin(PIN_R), RPiGPIOPin(PIN_G), RPiGPIOPin(PIN_B), active_high=True)
            led.blink(.25, .25, on_color=(1, 1, 1), n=5)
        except (OSError, RuntimeError, ImportError) as e:
            log('** Not using RGBLED with GPIOZERO ({} [{}]). Check your "{}" file.'
                .format(e, e.__class__, CONFIG_FILENAME), 'warn', verbose)
    return led


def led_alarm(led, time_blinking=2.5, timeout=3, time_on=.25, color=(1, 0, 0)):
    """Blinks for alarm code (one time or multiple times, with ∆T=.25 sec or 'time_on')"""
    if _use_led():
        if timeout == 0:
            led.blink(time_on, time_on, on_color=color)
        else:
            led.blink(time_on, time_on, on_color=color, n=int(time_blinking / (2 * time_on)))


def led_info(led, n=3):
    """Blinks in BLUE n times with ∆T=.5 sec"""
    if _use_led():
        led.blink(.5, .5, on_color=(0, 0, 1), n=n)


def blink_color(led, color, n=1):
    """Blinks n times in color with ∆T=.5 sec"""
    if _use_led():
        led.blink(.5, .5, on_color=color, n=n)


