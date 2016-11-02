**********
**ENERPI**
**********

⚡⚡ AC Current Meter for Raspberry PI ⚡⚡
=======================================

A simple current meter based on:

- **SCT-030 030** Current sensor,
- **MCP3008** Analog to Digital converter (on RASP.IO Analog Zero Hat)
- **Raspberry PI**

* Plus a little web server (**flask** based) with real-time streaming, beautiful **matplotlib** svg's,
and some **bokeh** plots.

.. Warning:: TODO Hacer descripción del módulo

-------------------------------


To get started:

.. code:: bash

    pip install enerpi

    enerpi -h


CLI Help:

.. code::

    usage: enerpi [-h] [-e] [-r] [-d] [-f [TS]] [-p [IM]] [--store ST] [--compact]
              [--backup BKP] [--clear] [--clearlog] [-i] [--last] [--temps]
              [-l] [--debug] [-v] [-T ∆T] [-ts ∆T] [-w ∆T]

    ⚡⚡ ︎ENERPI AC CURRENT SENSOR ⚡⚡

    AC Current Meter for Raspberry PI with GPIOZERO and MCP3008

    optional arguments:
      -h, --help            show this help message and exit

    ☆  ENERPI Working Mode:
      →  Choose working mode between RECEIVER / SENDER

      -e, -s, --enerpi      ⚡  SET ENERPI LOGGER & BROADCAST MODE
      -r, --receive         ⚡  SET Broadcast Receiver mode (by default)
      -d, --demo            ☮️  SET Demo Mode (broadcast random values)

    ℹ️  QUERY & REPORT DATA:
      -f [TS], --filter [TS]
                        ✂️  Query the HDF Store with pandas-like slicing:
                             "2016-01-07 :: 2016-02-01 04:00" --> df.loc["2016-01-07":"2016-02-01 04:00"]
                             (Pay atention to the double "::"!!)
                             · By default, "-f" filters data from 24h ago (.loc[2016-08-07 17:14:48:]).

      -p [IM], --plot [IM]  ⎙  Plot & save image with matplotlib in any compatible format.
                             · If not specified, PNG file is generated with MASK:
                                   "enerpi_potencia_consumo_ldr_{{:%Y%m%d_%H%M}}_{{:%Y%m%d_%H%M}}.png" using datetime data limits.
                             · If only specifying image format, default mask is used with the desired format.
                             · If image path is passed, the initial (and final, optionally) timestamps of filtered data
                             can be used with formatting masks, like:
                                 "/path/to/image/image_{:%c}_{:%H%M}.pdf" or "report_{:%d%m%y}.svg".

    ⚙  HDF Store Options:
      --store ST            ✏️  Set the .h5 file where save the HDF store.
                             Default: "/Users/uge/Dropbox/PYTHON/PYPROJECTS/enerpi/enerpi/../DATA/enerpi_data.h5"
      --compact             ✙✙  Compact the HDF Store database (read, delete, save)
      --backup BKP          ☔️  Backup the HDF Store
      --clear               ☠  Delete the HDF Store database
      --clearlog            ⚠️  Delete the LOG FILE at: "/Users/uge/Dropbox/PYTHON/PYPROJECTS/enerpi/enerpi/../DATA/enerpi.log"
      -i, --info            ︎ℹ️  Show data info
      --last                ︎ℹ️  Show last saved data

    ☕  DEBUG Options:
      --temps               ♨️  Show RPI temperatures (CPU + GPU)
      -l, --log             ☕  Show LOG FILE
      --debug               ☕  DEBUG Mode (save timing to csv)
      -v, --verbose         ‼️  Verbose mode ON BY DEFAULT!

    ⚒  Current Meter Sampling Configuration:
      -T ∆T, --delta ∆T     ⌚  Set Ts sampling (to database & broadcast), in seconds. Default ∆T: 1 s
      -ts ∆T                ⏱  Set Ts raw sampling, in ms. Default ∆T_s: 12 ms
      -w ∆T, --window ∆T    ⚖  Set window width in seconds for instant RMS calculation. Default ∆T_w: 2 s

    *** By default, ENERPI starts as receiver (-r) ***

============ =============
|left-image| |right-image|
============ =============

|plot-image|


.. code::

    ⚡ ︎ENERPI AC CURRENT SENSOR ⚡⚡
       AC Current Meter for Raspberry PI with GPIOZERO and MCP3008
       SENDER - RECEIVER vía UDP. Broadcast IP: 192.168.1.255, PORT: 57775
    ⚡ 17:10:51.380: 378 W; LDR=0.546 ◼◼◼◼◼◼◼◼◼◼◼◼︎⇡


.. |left-image| image:: https://github.com/azogue/enerpi/blob/master/docs/screenshot_cli_enerpi%20local%20receiver.png?raw=true
       :width: 100%
       :alt: CLI Receiver
       :align: bottom

.. |right-image| image:: https://github.com/azogue/enerpi/blob/master/docs/screenshot_cli_enerpi%20compact%2C%20backup%2C%20clear%20store.png?raw=true
       :width: 100%
       :alt: CLI Compact & Backup data
       :align: bottom

.. |plot-image| image:: https://github.com/azogue/enerpi/blob/master/docs/sample_plot_power_consumption_ldr.png?raw=true
       :width: 120%
       :alt: Matplotlib plot sample
       :align: bottom
