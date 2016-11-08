# -*- coding: utf-8 -*-
from collections import OrderedDict
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


def ppdict(dict_to_print, br='\n', html=False, key_align='l', sort_keys=True,
           key_preffix='', key_suffix='', value_prefix='', value_suffix='', left_margin=3, indent=2):
    if dict_to_print:
        if sort_keys:
            dic = dict_to_print.copy()
            keys = list(dic.keys())
            keys.sort()
            dict_to_print = OrderedDict()
            for k in keys:
                dict_to_print[k] = dic[k]

        tmp = ['{']
        ks = [type(x) == str and "'%s'" % x or x for x in dict_to_print.keys()]
        vs = [type(x) == str and "'%s'" % x or x for x in dict_to_print.values()]
        max_key_len = max([len(str(x)) for x in ks])

        for i in range(len(ks)):
            k = {1: str(ks[i]).ljust(max_key_len),
                 key_align == 'r': str(ks[i]).rjust(max_key_len)}[1]

            v = vs[i]
            tmp.append(' ' * indent + '{}{}{}:{}{}{},'.format(key_preffix, k, key_suffix,
                                                              value_prefix, v, value_suffix))

        tmp[-1] = tmp[-1][:-1]  # remove the ',' in the last item
        tmp.append('}')

        if left_margin:
            tmp = [' ' * left_margin + x for x in tmp]

        if html:
            return '<code>{}</code>'.format(br.join(tmp).replace(' ', '&nbsp;'))
        else:
            return br.join(tmp)
    else:
        return '{}'
