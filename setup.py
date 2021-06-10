# -*- coding: utf-8 -*-
"""Setup the package."""
from setuptools import setup

tests_require = [
    "pyuvsim @ git+git://github.com/RadioAstronomySoftwareGroup/pyuvsim",
]

setup(tests_require=tests_require)
