# -*- coding: utf-8 -*-
from pitemps import *


def main():
    print("CPU temp: ", str(get_cpu_temp()))
    print("GPU temp: ", str(get_gpu_temp()))


if __name__ == '__main__':
    main()
