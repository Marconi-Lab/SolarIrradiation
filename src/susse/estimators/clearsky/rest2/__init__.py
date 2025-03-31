# -*- coding: utf-8 -*-
"""
REST2 package
"""
from rest2 import *
import os
from .version import __version__
from .rest2 import rest2 
__all__ = ['rest2']


REST2DIR = os.path.dirname(os.path.realpath(__file__))
