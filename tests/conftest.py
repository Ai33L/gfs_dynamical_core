"""Make sibling test helper modules importable (spectral_converter,
fortran_loader, shtns_reference)."""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
