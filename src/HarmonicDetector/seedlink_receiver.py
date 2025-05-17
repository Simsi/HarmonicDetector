from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.core.stream import Stream
import threading
import numpy as np


class SeedlinkReceiver(EasySeedLinkClient):
    """
    SeedlinkReceiver подключается к SeedLink-серверу,
    получает сейсмические пакеты по заданному сетевому коду,
    коду станции и шаблону каналов, и хранит данные
    каждого канала в отдельном буфере.
    Если verbose=True, будет печатать в консоль статусные сообщения.
    """

    def __init__(self, host: str, port: int,
                 network: str, station: str, selector: str = "CX?",
                 verbose: bool = False):
        server_url = f"{host}:{port}"
        super().__init__(server_url, autoconnect=False)
        self.server_url = server_url
        self.network = network
        self.station = station
        self.selector = selector
        self.verbose = verbose
        # Буферы для «сырых» данных (останутся для совместимости)
        self.buffers = {}
        self.metadata = {}
        # Streams для правильной синхронизации по времени
        self.streams = {}      # { 'CX1': Stream([...]), ... }
        self._thread = None

    def connect(self):
        if self.verbose:
            print(f"[SeedlinkReceiver] Connecting to {self.server_url}…")
        super().connect()
        self.select_stream(self.network, self.station, self.selector)
        if self.verbose:
            print(f"[SeedlinkReceiver] Connected and subscribed to "
                  f"{self.network}.{self.station}.{self.selector}")

    def _on_data(self, trace):
        ch = trace.stats.channel
        # 1) Старый буфер:
        self.buffers.setdefault(ch, []).append(trace.data)
        self.metadata[ch] = {
            'starttime':     trace.stats.starttime,
            'sampling_rate': trace.stats.sampling_rate,
            'npts':          trace.stats.npts
        }
        # 2) Новый поток для временной синхронизации:
        if ch not in self.streams:
            self.streams[ch] = Stream()
        self.streams[ch] += trace

        if self.verbose:
            print(f"[{trace.stats.starttime.isoformat()}] "
                  f"Received {ch}: {trace.stats.npts} samples "
                  f"(buffer pkts: {len(self.buffers[ch])})")

    def run(self):
        self.on_data = self._on_data
        target_fn = lambda: super(SeedlinkReceiver, self).run()
        self._thread = threading.Thread(target=target_fn, daemon=True)
        if self.verbose:
            print("[SeedlinkReceiver] Starting receiver thread…")
        self._thread.start()

    def get_buffers(self):
        return self.buffers

    def get_metadata(self):
        return self.metadata

    def get_current_data(self):
        """
        Возвращает dict {
            'signal': ndarray shape (3, L),
            'metadata': {
                'starttime': UTCDateTime,
                'sampling_rate': float,
                'npts': int
            }
        }
        где L = число сэмплов в общем пересечении временных окон всех трёх каналов.
        """
        if len(self.streams) < 3:
            raise RuntimeError("Ещё недостаточно данных: получено каналов: "
                               f"{list(self.streams)}")

        # Объединяем пакеты в один Trace на канал
        merged_traces = {}
        for ch, st in self.streams.items():
            st_copy = st.copy()
            st_copy.merge(fill_value=0)
            if len(st_copy) != 1:
                raise RuntimeError(f"После merge в потоке {ch} "
                                   f"получилось {len(st_copy)} записей")
            merged_traces[ch] = st_copy[0]

        # Общие границы
        starts = [tr.stats.starttime for tr in merged_traces.values()]
        ends   = [tr.stats.endtime   for tr in merged_traces.values()]
        t_start = max(starts)
        t_end   = min(ends)
        if t_end <= t_start:
            raise RuntimeError("Нечего синхронизировать: пересечения нет")

        # Обрезаем и собираем данные
        data_arrays = []
        for ch in sorted(merged_traces):
            tr = merged_traces[ch].copy().trim(t_start, t_end)
            data_arrays.append(tr.data)

        signal = np.vstack(data_arrays)  # shape (3, L)
        metadata = {
            'starttime':     t_start,
            'sampling_rate': merged_traces[next(iter(merged_traces))].stats.sampling_rate,
            'npts':           signal.shape[1]
        }
        return {'signal': signal, 'metadata': metadata}


if __name__ == "__main__":
    receiver = SeedlinkReceiver(
        host="192.168.1.5",
        port=18000,
        network="XX",
        station="18",
        selector="CX?",
        verbose=False
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