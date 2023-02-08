# -*- coding: utf-8 -*-
import sympl

from ._components import (GFSDynamicalCore)


sympl.set_constant('top_of_model_pressure', 20., 'Pa')

__all__ = (GFSDynamicalCore)

__version__ = '0.16.14'
