# -*- coding: utf-8 -*-
from codecs import open
import os
from setuptools import setup, find_packages
from enerpi import __version__ as version, BASE_PATH


# Failed building wheel for cryptography, cffi
# Si error de cffi en install:
# sudo apt-get install build-essential libssl-dev libffi-dev python-dev

packages = find_packages(exclude=['docs', '*tests*', 'notebooks'])
with open(os.path.join(BASE_PATH, '..', 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='enerpi',
    version=version,
    description='AC Current Meter for Raspberry PI with GPIOZERO and MCP3008',
    long_description='\n' + long_description,
    keywords='enerpi current gpiozero raspberry analog',
    author='Eugenio Panadero',
    author_email='azogue.lab@gmail.com',
    url='https://github.com/azogue/enerpi',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Topic :: Home Automation',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Natural Language :: Spanish',
        'License :: OSI Approved :: MIT License',
        # 'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Operating System :: Unix'],
    packages=packages,
    package_data={
        'enerpi': ['config/default_config_enerpi.ini', 'config/default_sensors_enerpi.json',
                   'config/.enerpi_data_path'],
        'enerpiweb': ['templates/*', 'templates/email/*', 'templates/doc/*', 'templates/macros/*',
                      'static/css/*', 'static/img/icons/*', 'static/js/*'],
    },
    install_requires=['numpy>=1.11.2', 'pandas>=0.19.0', 'pytz>=2016.7',
                      'flask>=0.11.1', 'werkzeug>=0.11.11', 'Flask-Mail>=0.9.1', 'flask-autodoc>=0.1.2',
                      'Flask-WTF>=0.13.1', 'wtforms>=2.1',
                      'cryptography>=1.5.2',
                      'gpiozero>=1.3.1',  # 'RPi.GPIO>=0.6.3',
                      'matplotlib>=1.5.3', 'bokeh>=0.12.3', 'cairosvg>=1.0.22', 'termcolor>=1.1.0',
                      'python-crontab', 'pushbullet.py>=0.10.0', 'jsondiff>=1.0.0'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },
    entry_points={
        'console_scripts': [
            'enerpi = enerpi.__main__:enerpi_main_cli',
            'pitemps = enerpi.pitemps.__main__:main',
            'enerpi-daemon = enerpi.enerdaemon:enerpi_daemon',
            'enerpi-rscgen= enerpiplot.mule_rscgen:main',
            'enerpiweb = enerpiweb.__main__:main'
        ]
    },
    # Test configuration:
    # setup_requires=['pytest-runner'],
    tests_require=['pytest>=3.0.0', 'freezegun>=0.3.8'],
    # test_suite='nose.collector',
)
