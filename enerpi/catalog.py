# -*- coding: utf-8 -*-
import pandas as pd
from enerpi.hdftscat import HDFTimeSeriesCatalog
from enerpi.base import log


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

DELTA_MAX_SUMMARY_BFILL = 120  # pd.Timedelta('2min')


def _compress_data(data, sensors):
    """
    Down-casting of raw data to minimize disk usage.

    RMS sensor data is reduced to float32
    MEAN sensor data take integer values from 0 to 1000 (int16)
    Ref counters of rms & mean sensors also take integer values (int16)
    :param data: :pd.DataFrame: raw data
    :return: :pd.DataFrame: compressed raw data
    """
    if (data is not None) and not data.empty:
        c1, c2, c3 = sensors.columns_sensors_rms, sensors.columns_sensors_mean, [sensors.ref_rms, sensors.ref_mean]
        try:
            data_c = data[c1 + c2 + c3].copy()
        except KeyError as e:
            c1, c2, c3 = [list(filter(lambda x: x in data, c)) for c in [c1, c2, c3]]
            log('KeyError in _compress_data: {}; existent columns: {}'.format(e, [c1, c2, c3]), 'error', False)
            data_c = data[c1 + c2 + c3].copy()
        for c in c1:
            data_c[c] = data_c[c].astype('float32')
        for c in c2:
            if data_c[c].dtype != 'int16':
                data_c[c] = pd.Series(1000. * data_c[c]).round().astype('int16')
        for c in c3:
            data_c[c] = data_c[c].astype('int16')
        return data_c
    return data


def _process_data(data, sensors):
    """
    Calculate summary data (consumption and some stats of enerpi data...)

    :param data: :pd.DataFrame: archived data
    :return:
    """
    if data is not None and not data.empty:
        data = data.copy()
        col_delta = 'delta'
        col_cons = 'kWh'
        data[col_delta] = pd.Series(data.index).diff().fillna(method='bfill').dt.total_seconds().values
        data.loc[data[col_delta] > DELTA_MAX_SUMMARY_BFILL, col_delta] = DELTA_MAX_SUMMARY_BFILL
        cols_summary = []
        for i, c in enumerate(sensors.columns_sensors_rms):
            if i == 0:
                col_s = col_cons
            else:
                col_s = '{}_{}'.format(col_cons, c)
            cols_summary.append(col_s)
            data[col_s] = data[c] * data.delta / 3600
        cols_rs = list(sensors.columns_sensors_rms) + cols_summary + [col_delta]
        resampler = data[cols_rs].resample('1h', label='left')
        consumo = resampler[cols_summary].sum().fillna(0.).astype('float32')
        consumo /= 1000.
        consumo['t_ref'] = pd.Series(resampler[col_delta].sum() / 3600).astype('float32')
        consumo['p_max'] = resampler[sensors.main_column].max().round(0).astype('float16')
        consumo['p_mean'] = resampler[sensors.main_column].mean().round(0).astype('float16')
        consumo['p_min'] = resampler[sensors.main_column].min().round(0).astype('float16')
        data.drop([col_delta, col_cons], axis=1, inplace=True)
        return consumo
    return None


class EnerpiCatalog(HDFTimeSeriesCatalog):
    """
    ENERPI Catalog class for handling ENERPI Logger sampled data
    Based on HDFTimeSeriesCatalog

    """

    def __init__(self, sensors=None, **kwargs):
        if sensors is None:
            from enerpi.base import reload_config
            reload_config()
            from enerpi.base import SENSORS
            sensors = SENSORS
        self.sensors = sensors
        HDFTimeSeriesCatalog.__init__(self, **kwargs)

    @staticmethod
    def is_raw_data(data):
        """
        Test dataframe if its data is not compressed (if it's using default float64 dtypes)

        :param pd.DataFrame data: dataframe
        :return: is raw data
        :rtype: bool

        """
        for dt in data.dtypes:
            if dt == 'float64':
                return True
        return False

    def process_data(self, data):
        """
        Compress raw sampled data down-casting its dtypes before archive it in disk

        :param pd.DataFrame data: input data
        :return: compressed_data
        :rtype: pd.DataFrame

        """
        return _compress_data(data, self.sensors)

    def process_data_summary(self, data):
        """
        Process summary of data (calculate consumption data)

        :param pd.DataFrame data: input data to summarize
        :return: summary_data
        :rtype: pd.DataFrame

        """
        return _process_data(data, self.sensors)

    def process_data_summary_extra(self, data):
        """
        Process extra summary of data (calculate consumption data & report data -> Not implemented)

        :param pd.DataFrame data: input data
        :return: summary_data, extra_summary_data
        :rtype: tuple

        """
        # TODO Report data
        consumo = _process_data(data, self.sensors)
        return consumo, None
