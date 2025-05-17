#!/usr/bin/env python3
import numpy as np
from obspy import UTCDateTime
from HarmonicDetector import HarmonicDetector

def quick_test():
    # 1) Параметры синтетического сигнала
    fs = 500                       # Гц
    duration = 5.0                 # сек
    t = np.arange(0, duration, 1/fs)

    f1 = np.random.randint(200)
    f2 = np.random.randint(200)
    f3 = np.random.randint(200)
    print(f"Сгенерированы сигналы с частотами: {f1}, {f2}, {f3}")
    # 2) Три канала с разными частотами
    ch1 = np.sin(2*np.pi* f1 * t) + 0.05 * np.random.randn(len(t))
    ch2 = np.sin(2*np.pi* f2 * t) + 0.05 * np.random.randn(len(t))
    ch3 = np.sin(2*np.pi* f3 * t) + 0.05 * np.random.randn(len(t))
    signal3 = np.vstack([ch1, ch2, ch3])  # shape (3, L)

    # Абсолютное время старта
    start_time = UTCDateTime()

    # 3) Параметры детектора (подберите под ваш сигнал)
    h_bins            = 5
    Q_thr             = 2.0
    min_duration      = 0.5        # сек
    nperseg           = 256
    noverlap          = 128
    height_factor     = 1.0
    prominence_factor = 1.0
    min_sep_hz        = 10.0
    merge_jitter      = 10        # в отсчётах

    # 4) Создаём экземпляр
    detector = HarmonicDetector(
        fs=fs,
        h_bins=h_bins,
        Q_threshold=Q_thr,
        min_duration=min_duration,
        nperseg=nperseg,
        noverlap=noverlap,
        height_factor=height_factor,
        prominence_factor=prominence_factor,
        min_sep_hz=min_sep_hz,
        merge_jitter=merge_jitter
    )

    # 5) Запускаем инстанс-метод
    inst_preds = detector.detect(signal3, start_time)
    print("\n=== Instance method detect() results ===")
    if not inst_preds:
        print("  (none)")
    else:
        for freq, t0, t1 in inst_preds:
            print(f"  {freq:.1f} Hz: {t0.isoformat()} – {t1.isoformat()}")

    # 6) Запускаем статический метод
    static_preds = HarmonicDetector.detect_static(
        signal_3ch=signal3,
        start_time=start_time,
        fs=fs,
        h_bins=h_bins,
        Q_threshold=Q_thr,
        min_duration=min_duration,
        nperseg=nperseg,
        noverlap=noverlap,
        height_factor=height_factor,
        prominence_factor=prominence_factor,
        min_sep_hz=min_sep_hz,
        merge_jitter=merge_jitter
    )
    print("\n=== Static method detect_static() results ===")
    if not static_preds:
        print("  (none)")
    else:
        for freq, t0, t1 in static_preds:
            print(f"  {freq:.1f} Hz: {t0.isoformat()} – {t1.isoformat()}")

if __name__ == "__main__":
    quick_test()
