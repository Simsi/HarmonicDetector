import time
from obspy import UTCDateTime
from HarmonicDetector import HarmonicDetector, SeedlinkReceiver


if __name__ == "__main__":
    receiver = SeedlinkReceiver(
        host="192.168.1.5",
        port=18000,
        network="XX",
        station="18",
        selector="CX?",
        verbose=True,
    )
    # Параметры анализа
    fs = 500
    h_bins = 5
    Q_thr = 5.0
    min_sep_hz = 3
    height_fact = 2.0
    prom_fact = 2.0
    nperseg = fs
    noverlap = int(fs * 0.95)
    min_duration = 3
    merge_jitter_length = fs * 0.01

    # Инициализируем детектор
    detector = HarmonicDetector(
        fs=fs,
        h_bins=h_bins,
        Q_threshold=Q_thr,
        min_duration=min_duration,
        nperseg=nperseg,
        noverlap=noverlap,
        height_factor=height_fact,
        prominence_factor=prom_fact,
        min_sep_hz=min_sep_hz,
        merge_jitter=merge_jitter_length
    )

    receiver.connect()
    receiver.run()

    lasttime = datetime.datetime.now()
    interval = datetime.timedelta(seconds=1)

    try:
        while True:
            if datetime.datetime.now() - lasttime < interval:
                continue
            lasttime = datetime.datetime.now()

            data = receiver.get_current_data()
            raw3 = data["signal"]          # ndarray (3, L)
            meta = data["metadata"]
            global_start = meta["starttime"]

            # Собираем последний сегмент
            segment_len = int(fs * 2 * min_duration)
            offset = raw3.shape[1] - segment_len
            segment_start = global_start + offset / fs
            segment3 = raw3[:, -segment_len:]

            # --- 1) Метод экземпляра ---
            inst_preds = detector.detect(segment3, segment_start)
            print("\n>>> Results from instance method detect:")
            if not inst_preds:
                print("  (None)")
            else:
                for freq, t0, t1 in inst_preds:
                    print(f"  {freq:.1f} Hz: {t0.isoformat()} – {t1.isoformat()}")

            # --- 2) Статический метод ---
            static_preds = HarmonicDetector.detect_static(
                signal_3ch=segment3,
                start_time=segment_start,
                fs=fs,
                h_bins=h_bins,
                Q_threshold=Q_thr,
                min_duration=min_duration,
                nperseg=nperseg,
                noverlap=noverlap,
                height_factor=height_fact,
                prominence_factor=prom_fact,
                min_sep_hz=min_sep_hz,
                merge_jitter=merge_jitter_length
            )
            print("\n>>> Results from static method detect_static:")
            if not static_preds:
                print("  (None)")
            else:
                for freq, t0, t1 in static_preds:
                    print(f"  {freq:.1f} Hz: {t0.isoformat()} – {t1.isoformat()}")

            print("\n" + "-"*50 + "\n")

    except KeyboardInterrupt:
        if receiver.verbose:
            print("\n[SeedlinkReceiver] Stopped by user")