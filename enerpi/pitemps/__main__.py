# -*- coding: utf-8 -*-
"""
PITEMPS - __main__ module for CL use

"""
from enerpi.pitemps import get_cpu_temp, get_gpu_temp


def main():
    """
    Prints RPI temperatures on sys.stdout.

    """
    print("CPU temp: ", str(get_cpu_temp()))
    print("GPU temp: ", str(get_gpu_temp()))


if __name__ == '__main__':
    main()
