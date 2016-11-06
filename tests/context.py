# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.stats import norm, multivariate_normal
from numpy import linspace
from densratio import densratio
