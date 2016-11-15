# -*- coding: utf-8 -*-
# from http://doc.pytest.org/en/latest/example/simple.html - incremental testing - test steps
import pytest
import os
import shutil
import tempfile
from unittest.mock import patch
from enerpi import BASE_PATH


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed ({})".format(previousfailed.name))


def get_temp_catalog_for_testing(subpath_test_files='test_context_enerpi', check_integrity=True):
    """
    Copy example ENERPI files & sets common data catalog for testing.

    """
    dir_config = os.path.join(BASE_PATH, 'config')
    path_default_datapath = os.path.join(dir_config, '.enerpi_data_path')
    before_tests = open(path_default_datapath).read()

    # Prepara archivos:
    path_files_test = os.path.join(BASE_PATH, 'tests', 'rsc', subpath_test_files)
    tmp_dir = tempfile.TemporaryDirectory(prefix='ENERPIDATA_test')
    data_path = tmp_dir.name
    open(path_default_datapath, 'w').write(data_path)
    try:
        shutil.copytree(path_files_test, data_path)
    except FileExistsError:
        tmp_dir.cleanup()  # shutil.rmtree(data_path)
        shutil.copytree(path_files_test, data_path)
    with patch('builtins.input', return_value='1'):
        from enerpi.api import enerpi_data_catalog
        cat = enerpi_data_catalog(base_path=data_path, check_integrity=check_integrity, verbose=True)

    return tmp_dir, data_path, cat, path_default_datapath, before_tests
