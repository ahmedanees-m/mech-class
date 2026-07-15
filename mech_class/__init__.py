"""MECH-CLASS: Mechanism classification for programmable genome-writing enzymes.

Part of PEN-STACK. Builds on GENOME-ATLAS via genome-atlas>=0.7.2.
"""

try:
    from mech_class._version import __version__
except ImportError:
    __version__ = "unknown"

from mech_class.api import Prediction, Predictor

__all__ = ["__version__", "Predictor", "Prediction"]
