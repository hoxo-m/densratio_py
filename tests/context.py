import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../densratio/'))
sys.path.insert(0, path)

from densratio import densratio
import helpers
