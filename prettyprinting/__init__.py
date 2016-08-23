# -*- coding: utf-8 -*-
from termcolor import cprint


# Colorized output:
def print_info(x):
    cprint(x, 'blue')


def print_infob(x):
    cprint(x, 'blue', attrs=['bold'])


def print_ok(x):
    cprint(x, 'green', attrs=['bold'])


def print_secc(x):
    cprint(' ==> ' + x, 'blue', attrs=['bold', 'underline'])


def print_err(x):
    cprint('ERROR: ' + str(x), on_color='on_red', attrs=['bold'])


def print_warn(x):
    cprint('WARNING: ' + str(x), 'magenta')


def print_bold(x):
    cprint(x, attrs=['bold'])


def print_boldu(x):
    cprint(x, 'grey', attrs=['bold', 'underline'])


def print_yellowb(x):
    cprint(x, 'yellow', attrs=['bold'])


def print_grey(x):
    cprint(x, 'grey')


def print_greyb(x):
    cprint(x, 'grey', attrs=['bold'])


def print_red(x):
    cprint(x, 'red')


def print_redb(x):
    cprint(x, 'red', attrs=['bold'])  # en shared


def print_green(x):
    cprint(x, 'green')


def print_yellow(x):
    cprint(x, 'yellow')


def print_blue(x):
    cprint(x, 'blue')


def print_magenta(x):
    cprint(x, 'magenta')


def print_cyan(x):
    cprint(x, 'cyan')


def print_white(x):
    cprint(x, 'white')
