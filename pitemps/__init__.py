# -*- coding: utf-8 -*-
import subprocess


def get_cpu_temp():
    cmd = "/sys/class/thermal/thermal_zone0/temp"
    try:
        with open(cmd, 'r') as temp_file:
            cpu_temp = temp_file.read()
        return float(cpu_temp) / 1000
    except FileNotFoundError:
        print('Is this a RPI? "{}" not found!'.format(cmd))
        return None


def get_gpu_temp():
    try:
        gpu_temp = subprocess.check_output(['/opt/vc/bin/vcgencmd', 'measure_temp']
                                           ).decode().replace("temp=", "").replace("'C", "")
        return float(gpu_temp)
    except FileNotFoundError:
        print('Is this a RPI? "{}" not found!'.format('/opt/vc/bin/vcgencmd'))
        return None
