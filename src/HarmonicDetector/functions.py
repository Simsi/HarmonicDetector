import numpy as np
from scipy.signal import find_peaks, stft

def detect_harmonic_intervals_via_spectrogram(
    signal: np.ndarray,
    fs: float,
    h_bins: int,
    Q_threshold: float,
    min_duration: float,
    nperseg: int = 256,
    noverlap: int = 128,
    height_factor: float = 1.0,
    prominence_factor: float = 1.0,
    min_sep_hz: float = 5.0
) -> list[tuple]:
    """
    :returns: список (freq, t_start, t_end) участков, где
              Q[i,j] > Q_threshold подряд ≥ min_duration.
    """
    # 1) STFT
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg,
                     noverlap=noverlap, boundary=None)
    A = np.abs(Zxx)           # (M, N)
    M, N = A.shape
    dt = t[1] - t[0]
    min_frames = int(np.ceil(min_duration / dt))

    # 2) Подбросим пороги для peak-finder по среднему спектру
    avgA = A.mean(axis=1)
    med = np.median(avgA)
    sig = np.std(avgA)
    height_thd     = med + height_factor * sig
    prominence_thd = med + prominence_factor * sig
    df = f[1] - f[0]
    distance_bins  = int(np.ceil(min_sep_hz / df))

    # 3) Для каждой колонки j строим маску Q>threshold
    mask = np.zeros_like(A, dtype=bool)
    for j in range(N):
        col = A[:, j]
        # найти пики в этой колонке
        peaks_idx, _ = find_peaks(col,
                                  height=height_thd,
                                  prominence=prominence_thd,
                                  distance=distance_bins)
        for i in peaks_idx:
            # фон в freq-полосе вокруг i
            lo = max(i - h_bins, 0)
            hi = min(i + h_bins + 1, M)
            idx = np.r_[lo:i, i+1:hi]
            bg = A[idx, j].mean() if idx.size else 0
            Qij = col[i] / bg if bg>0 else 0
            mask[i, j] = (Qij > Q_threshold)

    # 4) По каждой частоте i ищем подряд True длиной ≥ min_frames
    intervals = []
    for i, fi in enumerate(f):
        row = mask[i]
        in_run = False
        run_start = 0
        for j, val in enumerate(row):
            if val and not in_run:
                in_run = True
                run_start = j
            elif (not val or j==N-1) and in_run:
                run_end = j+1 if (val and j==N-1) else j
                length = run_end - run_start
                if length >= min_frames:
                    t0 = t[run_start]
                    t1 = t[run_end-1] + dt
                    intervals.append((fi, t0, t1))
                in_run = False

    return intervals


def merge_close_intervals(
    intervals: list[tuple],
    total_duration: float,
    merge_jitter_length: float = 10
) -> list[tuple]:
    """
    Для каждой частоты объединяет подряд идущие интервалы,
    разрыв между которыми <= merge_ratio*total_duration.

    :param intervals:      список (freq, t0, t1)
    :param total_duration: общая длительность сигнала в секундах
    :param merge_jitter_length:    макс. разрыв для склейки в отсчётах
    :return:               список объединённых (freq, t0, t1)
    """
    if not intervals:
        return []

    max_gap = merge_jitter_length
    # сортируем сначала по частоте, потом по времени
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = []
    cur_freq, cur_t0, cur_t1 = intervals[0]

    for freq, t0, t1 in intervals[1:]:
        if freq == cur_freq and t0 - cur_t1 <= max_gap:
            # склеиваем
            cur_t1 = max(cur_t1, t1)
        else:
            merged.append((cur_freq, cur_t0, cur_t1))
            cur_freq, cur_t0, cur_t1 = freq, t0, t1

    # добавить последний
    merged.append((cur_freq, cur_t0, cur_t1))
    return merged