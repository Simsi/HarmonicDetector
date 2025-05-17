"""
HarmonicDetector — приём данных из SeedLink и детекция гармонических источников.
"""

__version__ = "0.1.0"

from .functions import (
    detect_harmonic_intervals_via_spectrogram,
    merge_close_intervals,
)
from .harmonic_detector import HarmonicDetector
from .seedlink_receiver import SeedlinkReceiver

__all__ = [
    "detect_harmonic_intervals_via_spectrogram",
    "merge_close_intervals",
    "HarmonicDetector",
    "SeedlinkReceiver"
]
