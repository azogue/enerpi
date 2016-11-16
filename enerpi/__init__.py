# -*- coding: utf-8 -*-
"""
* ︎ENERPI AC CURRENT SENSOR
* ANALOG SENSING with RASP.io ANALOG ZERO (Raspberry PI Hat with MCP3008) *

** ENERPI SCHEMATIC:

  ·············                              Rb:= Burden resistor (into the SCT030-030 probe, ~62 Ω)
  ·  ____________________↘○ ANALOG IN        C:= 10µF capacitor
  ·  │    │   ·                              R1, R2:= 2x 10KΩ resistor
  ·  #    ︎│   ·    _______• VREF (3.3V)
  ·  #   Rb   ·    │                         SCT probe connected to A_in (○) & V_divider (○)
  ·  #    │   ·    R1                          * Ideally, R1 = R2, so V_divider ≅ Vref / 2
  ·  │    │   ·    │
  ·  _____│_______↘○__R2__• GND
  ·············    │      │
    SCT030-030     │__||__│
                       C  │
                          ⏚

** LIGHT SENSOR SCHEMATIC:

   •____𝙍____•___LDR___•                    LDR:= Light Resistor
   │         │         │                     R:= 10KΩ resistor
   ⏚      ANALOG     VREF
   GND  	IN       (3.3V)

"""
import os


PRETTY_NAME = "︎⚡⚡ ︎ENERPI AC CURRENT SENSOR ⚡⚡"
DESCRIPTION = 'AC Current Meter for Raspberry PI with GPIOZERO and MCP3008'
BASE_PATH = os.path.abspath(os.path.dirname(__file__))

__version__ = '0.8.8.7'
