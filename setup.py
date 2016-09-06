# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from codecs import open
import os
from enerpi import VERSION


with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

PACKAGES = find_packages()

setup(
    # ext_modules=cythonize("extractlog.pyx"),
    name='enerpi',
    version=VERSION,
    description='AC Current Meter for Raspberry PI with GPIOZERO and MCP3008',
    long_description=LONG_DESCRIPTION,
    keywords='enerpi current gpiozero raspberry analog',
    # Author details
    author='Eugenio Panadero',
    author_email='azogue.lab@gmail.com',
    url='https://github.com/azogue/enerpi',

    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Topic :: Home Automation',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Natural Language :: Spanish',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Operating System :: Unix'
    ],
    packages=PACKAGES,
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'enerpi': ['rsc/paleta_power_w.csv', 'config_enerpi.ini', 'enerdaemon.sh'],
        'enerpiweb': ['enerpiweb.ini', 'uwsgi_mac.ini', 'enerpiweb_nginx.conf',
                      'templates/*', 'static/css/*', 'static/img/icons/*', 'static/css/*'],
    },

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'pandas', 'pytz', 'cryptography', 'gpiozero', 'matplotlib', 'termcolor', 'flask',
                      'werkzeug', 'jinja2', 'bokeh'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    entry_points={
        'console_scripts': [
            'enerpi = enerpi.__main__:enerpi_main_cli',
            'pitemps = pitemps.__main__:main',
            'enerpi-daemon = enerpi.enerdaemon:enerpi_daemon',
            'enerpiweb = enerpiweb.__main__:main'
        ]
    },
)
