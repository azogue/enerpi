# -*- coding: utf-8 -*-
import pandas as pd
from enerpi.hdftscat import HDFTimeSeriesCatalog
from enerpi.base import COLS_DATA


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
COL_POWER = COLS_DATA[0]
COL_REF = COLS_DATA[-2]
# TODO Terminar Doc


def _compress_data(data):
    if data is not None:
        if not data.empty:
            data = data.copy().astype('float32')
            data[COL_REF] = data[COL_REF].astype('int16')
            data['ldr'] = pd.Series(1000. * data['ldr']).round().astype('int16')
    return data


def process_data(data, append_consumo=False):
    consumo = None
    if data is not None and not data.empty and (append_consumo or ('high_delta' not in data.columns)):
        data = data.copy()
        data['delta'] = pd.Series(data.index).diff().fillna(method='bfill').dt.total_seconds().values
        data['high_delta'] = False
        data['execution'] = False
        data.loc[data['delta'] > 3, 'high_delta'] = True
        data.loc[data['delta'] > 60, 'execution'] = True
        data.loc[data['delta'] > DELTA_MAX_CALC_CONSUMO_SAMPLE_BFILL, 'delta'] = DELTA_MAX_CALC_CONSUMO_SAMPLE_BFILL
        data['Wh'] = data[COL_POWER] * data.delta / 3600
        if append_consumo:
            resampler = data[[COL_POWER, 'Wh', 'delta', 'high_delta', 'execution']].resample('1h', label='left')
            consumo = pd.DataFrame(resampler['Wh'].sum().rename('kWh')).fillna(0.).astype('float32')
            consumo /= 1000.
            consumo['t_ref'] = pd.Series(resampler['delta'].sum() / 3600).astype('float32')
            consumo['n_jump'] = resampler['high_delta'].sum().fillna(0).astype('int16')
            consumo['n_exec'] = resampler['execution'].sum().fillna(0).astype('int32')
            consumo['p_max'] = resampler[COL_POWER].max().round(0).astype('float16')
            consumo['p_mean'] = resampler[COL_POWER].mean().round(0).astype('float16')
            consumo['p_min'] = resampler[COL_POWER].min().round(0).astype('float16')
        data['high_delta'] = data['high_delta'].astype(bool)
        data['execution'] = data['execution'].astype(bool)
        data.drop(['delta', 'Wh'], axis=1, inplace=True)
        if append_consumo:
            return data, consumo
        return data
    elif append_consumo:
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
        return process_data(data, append_consumo=False)

    @staticmethod
    def process_data_summary(data):
        return process_data(data, append_consumo=True)

    @staticmethod
    def process_data_summary_extra(data):
        data, consumo = process_data(data, append_consumo=True)
        return data, consumo, None
