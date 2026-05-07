"""MECH-CLASS: Mechanism classification for programmable genome-writing enzymes.

Part of PEN-STACK (Paper 2).
Builds on GENOME-ATLAS (Paper 1) via genome-atlas>=0.6.0.
"""

try:
    from mech_class._version import __version__
except ImportError:
    __version__ = "unknown"

from mech_class.api import Predictor, Prediction

__all__ = ["__version__", "Predictor", "Prediction"]
