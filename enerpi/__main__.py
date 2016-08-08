# -*- coding: utf-8 -*-
import argparse
import datetime as dt
import logging
import os
from threading import Timer

from enerpi import PRETTY_NAME, DESCRIPTION, DATA_PATH, HDF_STORE, FILE_LOGGING, LOGGING_LEVEL
from enerpi.base import log, show_pi_temperature
from enerpi.database import (operate_hdf_database, get_ts_last_save, load_data, show_info_data,
                             extract_log_file, delete_log_file)
from enerpi.enerpimeter import (enerpi_logger, receiver, sender_random,
                                DELTA_SEC_DATA, TS_DATA_MS, RMS_ROLL_WINDOW_SEC)
from enerpi.enerplot import plot_potencia_consumo_horas, DEFAULT_IMG_MASK


def enerpi_arguments():
    """
    CLI Parser
    """
    p = argparse.ArgumentParser(description="\033[1m\033[5m\033[32m{}\033[0m\n{}\n\n".format(PRETTY_NAME, DESCRIPTION),
                                epilog='\033[34m\n*** By default, ENERPI starts as receiver (-r) ***\n' +
                                       '\033[0m', formatter_class=argparse.RawTextHelpFormatter)
    g_m = p.add_argument_group(title='☆  \033[1m\033[4mENERPI Working Mode\033[24m',
                               description='→  Choose working mode between RECEIVER / SENDER')
    g_m.add_argument('-e', '-s', '--enerpi', action='store_true', help='⚡  SET ENERPI LOGGER & BROADCAST MODE')
    g_m.add_argument('-r', '--receive', action='store_true', help='⚡  SET Broadcast Receiver mode (by default)')
    g_m.add_argument('-d', '--demo', action='store_true', help='☮️  SET Demo Mode (broadcast random values)')

    g_p = p.add_argument_group(title='︎ℹ️  \033[4mQUERY & REPORT DATA\033[24m')
    filter_24h = (dt.datetime.now().replace(microsecond=0) - dt.timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
    g_p.add_argument('-f', '--filter', action='store', nargs='?', metavar='TS', const=filter_24h,
                     help='✂️  Query the HDF Store with pandas-like slicing:'
                          '\n     "2016-01-07 :: 2016-02-01 04:00" --> df.loc["2016-01-07":"2016-02-01 04:00"]'
                          '\n     \t(Pay atention to the double "::"!!)'
                          '\n     · By default, "-f" filters data from 24h ago (.loc[{}:]).\n\n'.format(filter_24h))
    default_img_nomask = DEFAULT_IMG_MASK.replace('{', '{{').replace('}', '}}').replace('%', '%%')
    help_plot = '''⎙  Plot & save image with matplotlib in any compatible format.
     · If not specified, PNG file is generated with MASK:\n           "{}" using datetime data limits.
     · If only specifying image format, default mask is used with the desired format.
     · If image path is passed, the initial (and final, optionally) timestamps of filtered data
     can be used with formatting masks, like:
         "/path/to/image/image_{{:%%c}}_{{:%%H%%M}}.pdf" or "report_{{:%%d%%m%%y}}.svg".'''.format(default_img_nomask)
    g_p.add_argument('-p', '--plot', action='store', metavar='IM', nargs='?', const=DEFAULT_IMG_MASK, help=help_plot)

    g_st = p.add_argument_group(title='⚙  \033[4mHDF Store Options\033[24m')
    g_st.add_argument('--store', action='store', metavar='ST', default=HDF_STORE,
                      help='✏️  Set the .h5 file where save the HDF store.\n     Default: "{}"'.format(HDF_STORE))
    g_st.add_argument('--compact', action='store_true', help='✙✙  Compact the HDF Store database (read, delete, save)')
    g_st.add_argument('--backup', action='store', metavar='BKP', help='☔️  Backup the HDF Store')
    g_st.add_argument('--clear', action='store_true', help='☠  \033[31mDelete the HDF Store database\033[39m')
    g_st.add_argument('--clearlog', action='store_true', help='⚠️  Delete the LOG FILE at: "{}"'.format(FILE_LOGGING))
    g_st.add_argument('-i', '--info', action='store_true', help='︎ℹ️  Show data info')
    g_st.add_argument('--last', action='store_true', help='︎ℹ️  Show last saved data')

    g_d = p.add_argument_group(title='☕  \033[4mDEBUG Options\033[24m')
    g_d.add_argument('--temps', action='store_true', help='♨️  Show RPI temperatures (CPU + GPU)')
    g_d.add_argument('-l', '--log', action='store_true', help='☕  Show LOG FILE')
    g_d.add_argument('--debug', action='store_true', help='☕  DEBUG Mode (save timing to csv)')
    g_d.add_argument('-v', '--verbose', action='store_false', help='‼️  Verbose mode ON BY DEFAULT!')

    g_ts = p.add_argument_group(title='⚒  \033[4mCurrent Meter Sampling Configuration\033[24m')
    g_ts.add_argument('-T', '--delta', type=int, action='store', default=DELTA_SEC_DATA, metavar='∆T',
                      help='⌚  Set Ts sampling (to database & broadcast), in seconds. Default ∆T: {} s'
                      .format(DELTA_SEC_DATA))
    g_ts.add_argument('-ts', type=int, action='store', default=TS_DATA_MS, metavar='∆T',
                      help='⏱  Set Ts raw sampling, in ms. Default ∆T_s: {} ms'.format(TS_DATA_MS))
    g_ts.add_argument('-w', '--window', type=int, action='store', default=RMS_ROLL_WINDOW_SEC, metavar='∆T',
                      help='⚖  Set window width in seconds for instant RMS calculation. Default ∆T_w: {} s'
                      .format(RMS_ROLL_WINDOW_SEC))

    return p.parse_args()


def enerpi_main():
    # CLI Arguments
    args = enerpi_arguments()

    # CONTROL LOGIC
    if args.temps:
        # Shows RPI Temps
        Timer(3, show_pi_temperature, args=(3,)).start()

    if (args.enerpi or args.info or args.compact or args.backup or args.clear or
            args.last or args.clearlog or args.filter or args.plot):
        # Logging configuration
        logging.basicConfig(filename=FILE_LOGGING, level=LOGGING_LEVEL, datefmt='%d/%m/%Y %H:%M:%S',
                            format='%(levelname)s [%(filename)s_%(funcName)s] - %(asctime)s: %(message)s')
        log(PRETTY_NAME, 'ok', args.verbose)

        # Delete LOG File
        if args.clearlog:
            delete_log_file(FILE_LOGGING, verbose=True)

        # Data Store Config
        path_st = operate_hdf_database(args.store, compact=args.compact,
                                       path_backup=args.backup, clear_database=args.clear)

        # Starts ENERPI Logger
        if args.enerpi:
            # TODO Raw mode
            # TODO edit ∆T save_to_disk
            enerpi_logger(delta_sampling=args.delta, roll_time=args.window, sampling_ms=args.ts,
                          verbose=args.verbose, debug=args.debug, path_st=path_st)
        # Shows database info
        elif args.info or args.filter or args.plot:
            data, consumo = load_data(path_st, args.filter)
            if data is not None and args.info:
                show_info_data(data, consumo)
            if data is not None and args.plot:
                img_name = plot_potencia_consumo_horas(data.power, consumo, ldr=data.ldr,
                                                       rs_potencia=None, rm_potencia=None, savefig=args.plot)
                log('Imagen guardada en "{}"'.format(img_name), 'ok', args.verbose, True)
        # Shows database info
        else:  # Shows last 10 entries
            last = get_ts_last_save(path_st, get_last_sample=True, verbose=True, n=10)
            print(last)
    # Shows & extract info from LOG File
    elif args.log:
        data_log = extract_log_file(FILE_LOGGING, extract_temps=True, verbose=True)
        try:
            df_temps = data_log[data_log.temp.notnull()].dropna(axis=1).drop(['tipo', 'msg', 'debug_send'], axis=1)
            if len(set(df_temps['exec'])) == 1:
                df_temps = df_temps.drop(['exec'], axis=1)
            path_csv = os.path.join(DATA_PATH, 'debug_rpitemps.csv')
            if not df_temps.empty:
                df_temps.to_csv(path_csv)
                print('*** Grabados datos de temperatura extraidos del LOG en {}'.format(path_csv))
        except AttributeError:
            print('No se encuentran datos de Tª de RPI en el LOG')
    # Demo sender
    elif args.demo:
        path_st = os.path.join(DATA_PATH, 'debug_buffer_disk.h5')
        sender_random(ts_data=1, verbose=args.verbose, debug=args.debug, path_st=path_st)
    # Receiver
    else:  # elif args.receive:
        log('{}\n   {}'.format(PRETTY_NAME, DESCRIPTION), 'ok', args.verbose, False)
        receiver(verbose=args.verbose, debug=args.debug)


if __name__ == '__main__':
    enerpi_main()
