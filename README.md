# HarmonicDetector

## 

Алгоритм поиска устойчивых во времени высокоамплитудных частотных компонент сигнала.

## Установка

### Из репозитория
```bash
pip install git+https://github.com/Simsi/HarmonicDetector.git@develop
```

### Из исходников
```bash
git clone https://github.com/Simsi/HarmonicDetector --branch develop
cd HarmonicDetector
pip install .
```
## Описание

Алгоритм разработан для обнаружения в непрерывных сигналах устойчивых, выделяющихся на фоне шумов частотных компонент. <br>

### Параметры алгоритма
```python
fs = 500                  # частота дискретизации <br>
h_bins = 5                # кол-во бинов для алгоритма (значение отступа в Гц от обнаруженного пика частоты) <br>
Q_thr = 5.0               # пороговое значение отношения амплитуды пика к среднему из диапазона [peak_bin - h_bins; peak_bin + h_bins], не включая peak_bin. <br>
min_sep_hz = 3            # минимальный интервал между пиками (Гц) <для find_peaks> <br>
height_fact = 2.0         # сколько σ над медианой для height <для find_peaks> <br>
prom_fact = 2.0                      # сколько σ над медианой для prominence <для find_peaks> <br>
nperseg = fs                         # длина окна FFT <br>
noverlap = int(fs * 0.95)            # перекрытие окна FFT <br>
min_duration = 3                     # минимальная длина в секундах сигнала гармонического сигнала (в секундах) <br>
merge_jitter_length = int(fs * 0.01) # допустимый разрыв между обнаруженными интервалами для их объединения в один (в отчётах) <br>
```

### Входные данные алгоритма

На вход в алгоритм подаётся трёхканальный сигнал, <br>
обнаружения вычисляются для всех трёх каналов, <br>
после вычисления обнаружения объединяются в единый список. <br>
Если нужно произвести вычисление алгоритма на одноканальном сигнале, <br>
то можно дополнить каналами из нулей <br>
 
### Принцип работы алгоритма
1) Применение detrend к сигналу
2) Вычисление спектрограммы сигнала
3) Выполнить проход по сигналу по временной оси, анализируя спектры. <br>
  Анализ спектра происходит следующим образом: <br>
  а) Подбираем параметры для scipy.find_peaks по среднему в спектре и переданным параметрам <br>
  б) Определяем пики с помощью find_peaks <br>
  в) Смотрим для каждого F(peak_bin) из списка пиковых частот промежуток спектра close_bins = [peak_bin - h_bins; peak_bin + h_bins], находим отношение F(peak_bin) / avg_f(close_bins). <br>
  г) Все не пиковые частоты выставляются в 0, для всех пиковых считается пункт в) <br>
  д) Вычисляем маску для спектрограммы по пороговому значению Q, переданному пользователем <br>
  е) Определяем устойчивые промежутки, в которых сигнал держался выше порогового значения, длина которых больше указанной пользователем длины <br>
  ж) Возвращаем пользователю список обнаруженных промежутков в виде (freq, start_time, end_time) <br>

## Тестирование

### Готовые скрипты
Скрипты тестирования в папке examples.
Для тестирования без подключения к seiscomp серверу через seedlink используется quick_test.py. <br>
В этом скрипте генерируется сигнал фиксированной длины искусственно со случайными частотными компонентами. <br>

Для тестирования с подключением к seiscomp серверу через seedlink используется seedlink_example.py

### добавление в свой код
``` python
import numpy as np
from obspy import UTCDateTime
from HarmonicDetector import HarmonicDetector

# ... Ваш трёхканальный сигнал 
sig = np.zeros((3, 5000))

# Определение ваших параметров
h_bins            = 5
Q_thr             = 2.0
min_duration      = 0.5        # сек
nperseg           = fs
noverlap          = int(fs * 0.95)
height_factor     = 1.0
prominence_factor = 1.0
min_sep_hz        = 10.0
merge_jitter      = int(fs * 0.01)      # в отсчётах

# обнаружение статическим методом
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

if not static_preds:
   print("  (none)")
else:
   for freq, t0, t1 in static_preds:
       print(f"  {freq:.1f} Hz: {t0.isoformat()} – {t1.isoformat()}")

# Вывод следующего вида:
# >>> Results from static method detect_static:
#   90.0 Hz: 2025-05-17T04:44:25.158533 – 2025-05-17T04:44:30.208533


```


