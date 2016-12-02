# -*- coding: utf-8 -*-
"""
ENERPI tests - SENSORS configuration

"""
import enerpi.prettyprinting as pp
from tests.conftest import TestCaseEnerpi


class TestEnerpiCONFIG(TestCaseEnerpi):

    def test_sensors(self):
        """
        ENERPI CONFIG Testing. Analog Sensors

        """
        from enerpi.base import SENSORS

        pp.print_secc('REPR:')
        pp.print_yellowb('ANALOG SENSORS:\n{}'.format(SENSORS))

        for s in SENSORS:
            pp.print_red(s)
        print(SENSORS.ts_column, SENSORS.main_column, SENSORS.ts_fmt)
        print(SENSORS.columns_sensors, SENSORS.n_cols_sensors)
        print(SENSORS.columns_sampling, SENSORS.n_cols_sampling)
        print(SENSORS.descriptions(SENSORS.columns_sensors + SENSORS.columns_sampling))

        pp.print_secc('INCLUDED IN DATA:')
        data = {}
        cols_rms, cols_mean = SENSORS.included_columns_sensors(data)
        print(data, cols_rms, cols_mean)
        assert (len(cols_rms) == 0) and (len(cols_mean) == 0)

        data.update({'ts': '2016-01-01 08:03:05'})
        cols_rms, cols_mean = SENSORS.included_columns_sensors(data)
        print(data, cols_rms, cols_mean)
        assert (len(cols_rms) == 0) and (len(cols_mean) == 0)

        data.update({SENSORS.main_column: 33.3})
        cols_rms, cols_mean = SENSORS.included_columns_sensors(data)
        print(data, cols_rms, cols_mean)
        assert (len(cols_rms) == 1) and (len(cols_mean) == 0)

        data.update({SENSORS.columns_sensors[-1]: 44.3})
        cols_rms, cols_mean = SENSORS.included_columns_sensors(data)
        print(data, cols_rms, cols_mean)
        assert (len(cols_rms) == 1) and (len(cols_mean) == 1)

        pp.print_secc('ITER:')
        n = len(SENSORS)
        names = SENSORS.columns_sensors[1:]
        assert len(names) == n
        s_iterator = SENSORS.__iter__()
        counter = 0
        for name in names:
            try:
                sensor = next(s_iterator)
                pp.print_magenta(sensor)
                counter += 1
                assert sensor.name == name
            except StopIteration:
                assert counter == n

        pp.print_secc('GETATTR:')
        pp.print_info(SENSORS[0])
        pp.print_info(SENSORS[1])

        pp.print_cyan(SENSORS[SENSORS.columns_sensors_mean[0]])
        pp.print_cyan(SENSORS[SENSORS.columns_sensors_rms[0]])
        try:
            pp.print_cyan(SENSORS[99])
            assert 0
        except KeyError as e:
            pp.print_ok(e)
        try:
            pp.print_cyan(SENSORS['lalala'])
        except KeyError as e:
            pp.print_ok(e)


if __name__ == '__main__':
    import unittest

    unittest.main()
