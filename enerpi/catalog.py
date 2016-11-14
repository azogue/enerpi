# -*- coding: utf-8 -*-
import pandas as pd
from enerpi.hdftscat import HDFTimeSeriesCatalog
from enerpi.base import SENSORS


'''
    @staticmethod
    def process_data(data):
        # Implementar
        return data

    @staticmethod
    def process_data_summary(data):
        # Implementar
        return data, data

    @staticmethod
    def process_data_summary_extra(data):
        # Implementar
        return data, data, None
'''

DELTA_MAX_CALC_CONSUMO_SAMPLE_BFILL = 120  # pd.Timedelta('2min')
# TODO Terminar Doc


def _compress_data(data):
    """
    Down-casting of raw data to minimize disk usage.

    RMS sensor data is reduced to float32
    MEAN sensor data take integer values from 0 to 1000 (int16)
    Ref counters of rms & mean sensors also take integer values (int16)
    :param data: :pd.DataFrame: raw data
    :return: :pd.DataFrame: compressed raw data
    """
    if data is not None:
        if not data.empty:
            data = data.copy().astype('float32')
            data[SENSORS.ref_rms] = data[SENSORS.ref_rms].astype('int16')
            for c in SENSORS.columns_sensors_mean:
                data[c] = pd.Series(1000. * data[c]).round().astype('int16')
    return data


def _process_data(data, append_summary=False):
    """
    Calculate consumption and some stats of enerpi data...

    :param data: :pd.DataFrame: archived data
    :param append_summary: :bool: process summary of data and returns tuple of processed & summary dataframes
    :return:
    """
    consumo = None
    if data is not None and not data.empty and (append_summary or ('high_delta' not in data.columns)):
        data = data.copy()
        cols_rs = [SENSORS.main_column, 'Wh', 'delta', 'high_delta', 'execution']
        data[cols_rs[2]] = pd.Series(data.index).diff().fillna(method='bfill').dt.total_seconds().values
        data[cols_rs[3]] = False
        data[cols_rs[4]] = False
        data.loc[data[cols_rs[2]] > 3, cols_rs[3]] = True
        data.loc[data[cols_rs[2]] > 60, cols_rs[4]] = True
        data.loc[data[cols_rs[2]] > DELTA_MAX_CALC_CONSUMO_SAMPLE_BFILL,
                 cols_rs[2]] = DELTA_MAX_CALC_CONSUMO_SAMPLE_BFILL
        data[cols_rs[1]] = data[SENSORS.main_column] * data.delta / 3600
        if append_summary:
            resampler = data[cols_rs].resample('1h', label='left')
            consumo = pd.DataFrame(resampler[cols_rs[1]].sum().rename('kWh')).fillna(0.).astype('float32')
            consumo /= 1000.
            consumo['t_ref'] = pd.Series(resampler[cols_rs[2]].sum() / 3600).astype('float32')
            consumo['n_jump'] = resampler[cols_rs[3]].sum().fillna(0).astype('int16')
            consumo['n_exec'] = resampler[cols_rs[4]].sum().fillna(0).astype('int32')
            consumo['p_max'] = resampler[SENSORS.main_column].max().round(0).astype('float16')
            consumo['p_mean'] = resampler[SENSORS.main_column].mean().round(0).astype('float16')
            consumo['p_min'] = resampler[SENSORS.main_column].min().round(0).astype('float16')
        data[cols_rs[3]] = data[cols_rs[3]].astype(bool)
        data[cols_rs[4]] = data[cols_rs[4]].astype(bool)
        data.drop([cols_rs[2], cols_rs[1]], axis=1, inplace=True)
        if append_summary:
            return data, consumo
        return data
    elif append_summary:
        return data, None
    return data


class EnerpiCatalog(HDFTimeSeriesCatalog):

    @staticmethod
    def is_raw_data(data):
        if (len(data.columns) == 6) or (data['ldr'].dtype == 'int16'):
            return False
        return True

    @staticmethod
    def process_data(data, is_raw=True):
        if is_raw:
            data = _compress_data(data)
        return _process_data(data, append_summary=False)

    @staticmethod
    def process_data_summary(data):
        return _process_data(data, append_summary=True)

    @staticmethod
    def process_data_summary_extra(data):
        data, consumo = _process_data(data, append_summary=True)
        return data, consumo, None
