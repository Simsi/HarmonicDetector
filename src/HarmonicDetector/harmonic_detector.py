from obspy import UTCDateTime
from .functions import detect_harmonic_intervals_via_spectrogram, merge_close_intervals
from scipy.signal import detrend
import numpy as np


class HarmonicDetector:
    """
    Класс для детекции гармонических источников в многоканальном сигнале.
    Методы возвращают [(freq, abs_start, abs_end), …].
    """

    def __init__(self,
                 fs: float,
                 h_bins: int,
                 Q_threshold: float,
                 min_duration: float,
                 nperseg: int,
                 noverlap: int,
                 height_factor: float = 1.0,
                 prominence_factor: float = 1.0,
                 min_sep_hz: float = 5.0,
                 merge_jitter: float = 10.0):
        self.fs = fs
        self.h_bins = h_bins
        self.Q_thr = Q_threshold
        self.min_duration = min_duration
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.height_factor = height_factor
        self.prominence_factor = prominence_factor
        self.min_sep_hz = min_sep_hz
        self.merge_jitter = merge_jitter

    def detect(self,
               signal_3ch: np.ndarray,
               start_time: UTCDateTime
               ) -> list[tuple]:
        try:
            all_preds = []
            L = signal_3ch.shape[1]
            total_dur = L / self.fs

            for ch_idx in range(signal_3ch.shape[0]):
                # используем detrend из scipy.signal
                sig_ch = detrend(signal_3ch[ch_idx])

                intervals = detect_harmonic_intervals_via_spectrogram(
                    signal=sig_ch,
                    fs=self.fs,
                    h_bins=self.h_bins,
                    Q_threshold=self.Q_thr,
                    min_duration=self.min_duration,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    height_factor=self.height_factor,
                    prominence_factor=self.prominence_factor,
                    min_sep_hz=self.min_sep_hz
                )
                merged = merge_close_intervals(intervals, total_dur, self.merge_jitter)
                for freq, t0, t1 in merged:
                    all_preds.append((freq, start_time + t0, start_time + t1))

            # объединим по частоте пересекающиеся интервалы
            final = []
            preds_by_freq = {}
            for freq, t0, t1 in all_preds:
                preds_by_freq.setdefault(freq, []).append((t0, t1))
            for freq, ivs in preds_by_freq.items():
                ivs.sort(key=lambda x: x[0])
                cs, ce = ivs[0]
                for s, e in ivs[1:]:
                    if s <= ce:
                        ce = max(ce, e)
                    else:
                        final.append((freq, cs, ce))
                        cs, ce = s, e
                final.append((freq, cs, ce))

            final.sort(key=lambda x: (x[1], x[0]))
            return final

        except Exception as e:
            print("HarmonicDetector: " + str(e))
            return None

    @staticmethod
    def detect_static(signal_3ch: np.ndarray,
                      start_time: UTCDateTime,
                      fs: float,
                      h_bins: int,
                      Q_threshold: float,
                      min_duration: float,
                      nperseg: int,
                      noverlap: int,
                      height_factor: float = 1.0,
                      prominence_factor: float = 1.0,
                      min_sep_hz: float = 5.0,
                      merge_jitter: float = 10.0
                      ) -> list[tuple]:
        try:
            all_preds = []
            L = signal_3ch.shape[1]
            total_dur = L / fs

            for idx in range(signal_3ch.shape[0]):
                sig_ch = detrend(signal_3ch[idx])

                intervals = detect_harmonic_intervals_via_spectrogram(
                    signal=sig_ch,
                    fs=fs,
                    h_bins=h_bins,
                    Q_threshold=Q_threshold,
                    min_duration=min_duration,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    height_factor=height_factor,
                    prominence_factor=prominence_factor,
                    min_sep_hz=min_sep_hz
                )
                merged = merge_close_intervals(intervals, total_dur, merge_jitter)
                for freq, t0, t1 in merged:
                    all_preds.append((freq, start_time + t0, start_time + t1))

            final = []
            preds_by_freq = {}
            for freq, t0, t1 in all_preds:
                preds_by_freq.setdefault(freq, []).append((t0, t1))
            for freq, ivs in preds_by_freq.items():
                ivs.sort(key=lambda x: x[0])
                cs, ce = ivs[0]
                for s, e in ivs[1:]:
                    if s <= ce:
                        ce = max(ce, e)
                    else:
                        final.append((freq, cs, ce))
                        cs, ce = s, e
                final.append((freq, cs, ce))

            final.sort(key=lambda x: (x[1], x[0]))
            return final
        except Exception as e:
            print("HarmonicDetector: " + str(e))
            return None