import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy import signal

# Създаване на синтетични ЕКГ данни
def generate_synthetic_ekg(duration=10, sampling_rate=500, heart_rate=70):
    """
    Генерира синтетични ЕКГ данни
    
    Parameters:
    - duration: продължителност на ЕКГ записа в секунди
    - sampling_rate: честота на семплиране в Hz
    - heart_rate: сърдечен ритъм в удари в минута
    
    Returns:
    - времева ос
    - ЕКГ сигнал
    """
    # Изчисляване на интервала между ударите в секунди
    beat_interval = 60 / heart_rate  
    
    # Създаване на времева ос
    t = np.arange(0, duration, 1/sampling_rate)
    
    # Базова линия с малко шум
    baseline = 0.05 * np.sin(2 * np.pi * 0.25 * t) + 0.01 * np.random.randn(len(t))
    
    # Инициализиране на ЕКГ сигнала
    ekg = np.zeros_like(t)
    
    # Добавяне на PQRST компоненти към всяко сърдечно съкращение
    for beat_time in np.arange(0.5, duration, beat_interval):
        # P-вълна
        p_width = 0.08
        p_center = beat_time - 0.15
        p_wave = 0.15 * np.exp(-((t - p_center) ** 2) / (2 * p_width ** 2))
        
        # QRS комплекс
        q_center = beat_time - 0.05
        q_wave = -0.1 * np.exp(-((t - q_center) ** 2) / (2 * 0.01 ** 2))
        
        r_center = beat_time
        r_wave = 1.0 * np.exp(-((t - r_center) ** 2) / (2 * 0.01 ** 2))
        
        s_center = beat_time + 0.05
        s_wave = -0.2 * np.exp(-((t - s_center) ** 2) / (2 * 0.02 ** 2))
        
        # T-вълна
        t_width = 0.1
        t_center = beat_time + 0.25
        t_wave = 0.3 * np.exp(-((t - t_center) ** 2) / (2 * t_width ** 2))
        
        # Комбиниране на всички компоненти
        ekg += p_wave + q_wave + r_wave + s_wave + t_wave
    
    # Добавяне на базова линия и случаен шум
    ekg += baseline + 0.02 * np.random.randn(len(t))
    
    return t, ekg

# Функция за намиране на локални екстремуми (максимуми и минимуми)
def find_extrema(signal, threshold=0.1):
    """
    Намира локалните максимуми и минимуми на сигнала
    
    Parameters:
    - signal: ЕКГ сигнал
    - threshold: праг за откриване на пикове
    
    Returns:
    - местоположения на максимумите
    - местоположения на минимумите
    """
    # Намиране на локалните максимуми
    maxima, _ = find_peaks(signal, height=threshold)
    
    # Намиране на локалните минимуми чрез инвертиране на сигнала
    minima, _ = find_peaks(-signal, height=threshold)
    
    return maxima, minima

# Функция за изчисляване на първа и втора производна
def compute_derivatives(t, signal):
    """
    Изчислява първата и втората производна на ЕКГ сигнала
    
    Parameters:
    - t: времева ос
    - signal: ЕКГ сигнал
    
    Returns:
    - първа производна
    - втора производна
    """
    # Изчисляване на първата производна: df/dt
    first_derivative = np.gradient(signal, t)
    
    # Изчисляване на втората производна: d²f/dt²
    second_derivative = np.gradient(first_derivative, t)
    
    return first_derivative, second_derivative

# Функция за анализ на изпъкналост/вдлъбнатост
def analyze_concavity(second_derivative):
    """
    Анализира изпъкналост/вдлъбнатост на графиката базирано на втората производна
    
    Parameters:
    - second_derivative: втора производна на сигнала
    
    Returns:
    - индекси на изпъкналите части (втора производна < 0)
    - индекси на вдлъбнатите части (втора производна > 0)
    """
    # Изпъкнала част (втора производна < 0)
    convex_regions = np.where(second_derivative < 0)[0]
    
    # Вдлъбната част (втора производна > 0)
    concave_regions = np.where(second_derivative > 0)[0]
    
    return convex_regions, concave_regions

# Създаване на синтетични ЕКГ данни
t, ekg_signal = generate_synthetic_ekg(duration=5, heart_rate=70)

# Намиране на локални екстремуми
maxima_indices, minima_indices = find_extrema(ekg_signal, threshold=0.1)

# Изчисляване на производните
first_deriv, second_deriv = compute_derivatives(t, ekg_signal)

# Анализ на изпъкналост/вдлъбнатост
convex_indices, concave_indices = analyze_concavity(second_deriv)

# Визуализация на ЕКГ данните и ключовите компоненти
fig, axs = plt.subplots(3, 1, figsize=(15, 12))

# Плот на ЕКГ сигнала с отбелязани екстремуми
axs[0].plot(t, ekg_signal, 'b-', label='ЕКГ сигнал')
axs[0].plot(t[maxima_indices], ekg_signal[maxima_indices], 'ro', label='Локални максимуми')
axs[0].plot(t[minima_indices], ekg_signal[minima_indices], 'go', label='Локални минимуми')

# Добавяне на анотации за PQRST вълните
# Намиране на R-пикове (най-високите точки)
r_peaks = maxima_indices[np.argsort(ekg_signal[maxima_indices])[-5:]]
for i, peak in enumerate(r_peaks[:3]):  # Само за първите три за яснота
    # Намиране на Q и S точките около R-пика
    window_size = 50  # брой точки за търсене наоколо
    left_window = max(0, peak - window_size)
    right_window = min(len(ekg_signal), peak + window_size)
    
    q_idx = left_window + np.argmin(ekg_signal[left_window:peak])
    s_idx = peak + np.argmin(ekg_signal[peak:right_window])
    
    # Опит за откриване на P и T вълни
    p_window = max(0, q_idx - window_size)
    t_window = min(len(ekg_signal), s_idx + window_size)
    
    p_candidates = maxima_indices[(maxima_indices > p_window) & (maxima_indices < q_idx)]
    p_idx = p_candidates[0] if len(p_candidates) > 0 else None
    
    t_candidates = maxima_indices[(maxima_indices > s_idx) & (maxima_indices < t_window)]
    t_idx = t_candidates[0] if len(t_candidates) > 0 else None
    
    # Добавяне на анотации
    if p_idx:
        axs[0].annotate('P', (t[p_idx], ekg_signal[p_idx]), fontsize=12)
    axs[0].annotate('Q', (t[q_idx], ekg_signal[q_idx]), fontsize=12)
    axs[0].annotate('R', (t[peak], ekg_signal[peak]), fontsize=12)
    axs[0].annotate('S', (t[s_idx], ekg_signal[s_idx]), fontsize=12)
    if t_idx:
        axs[0].annotate('T', (t[t_idx], ekg_signal[t_idx]), fontsize=12)

axs[0].set_title('ЕКГ сигнал с отбелязани ключови точки', fontsize=14)
axs[0].set_xlabel('Време (s)', fontsize=12)
axs[0].set_ylabel('Амплитуда (mV)', fontsize=12)
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Плот на първата производна
axs[1].plot(t, first_deriv, 'r-', label='Първа производна')
axs[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[1].set_title('Първа производна на ЕКГ сигнала', fontsize=14)
axs[1].set_xlabel('Време (s)', fontsize=12)
axs[1].set_ylabel('dV/dt', fontsize=12)
axs[1].legend(loc='upper right')
axs[1].grid(True)

# Плот на втората производна с анализ на изпъкналост/вдлъбнатост
axs[2].plot(t, second_deriv, 'g-', label='Втора производна')
axs[2].fill_between(t, second_deriv, 0, where=(second_deriv < 0), color='red', alpha=0.3, label='Изпъкнала част (d²V/dt² < 0)')
axs[2].fill_between(t, second_deriv, 0, where=(second_deriv > 0), color='blue', alpha=0.3, label='Вдлъбната част (d²V/dt² > 0)')
axs[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[2].set_title('Втора производна и анализ на изпъкналост/вдлъбнатост', fontsize=14)
axs[2].set_xlabel('Време (s)', fontsize=12)
axs[2].set_ylabel('d²V/dt²', fontsize=12)
axs[2].legend(loc='upper right')
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Допълнителен анализ: Измерване на сърдечната честота от R-R интервалите
def calculate_heart_rate(t, ekg_signal, threshold=0.5):
    """
    Изчислява сърдечната честота от R-пиковете на ЕКГ сигнала
    
    Parameters:
    - t: времева ос
    - ekg_signal: ЕКГ сигнал
    - threshold: праг за откриване на R-пикове
    
    Returns:
    - средна сърдечна честота в удари в минута (BPM)
    - моментна сърдечна честота от всеки R-R интервал
    """
    # Намиране на R-пикове
    r_peaks, _ = find_peaks(ekg_signal, height=threshold)
    
    # Изчисляване на R-R интервалите във времеви единици
    rr_intervals = np.diff(t[r_peaks])
    
    # Изчисляване на моментната сърдечна честота за всеки R-R интервал (60 за преобразуване в минути)
    instant_hr = 60 / rr_intervals
    
    # Изчисляване на средната сърдечна честота
    mean_hr = np.mean(instant_hr)
    
    return mean_hr, instant_hr, r_peaks

# Изчисляване на сърдечната честота
mean_heart_rate, instant_heart_rate, r_peaks = calculate_heart_rate(t, ekg_signal, threshold=0.5)

# Визуализация на сърдечната честота
plt.figure(figsize=(12, 6))
plt.plot(t, ekg_signal, 'b-', label='ЕКГ сигнал')
plt.plot(t[r_peaks], ekg_signal[r_peaks], 'ro', markersize=8, label='R-пикове')

# Добавяне на текст със средната сърдечна честота
plt.text(0.5, max(ekg_signal) * 0.8, 
         f'Средна сърдечна честота: {mean_heart_rate:.1f} BPM', 
         fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

plt.title('Анализ на сърдечната честота от R-R интервалите', fontsize=16)
plt.xlabel('Време (s)', fontsize=14)
plt.ylabel('Амплитуда (mV)', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Визуализация на вариабилността на сърдечната честота
plt.figure(figsize=(12, 6))
plt.plot(t[r_peaks[:-1]], instant_heart_rate, 'r.-', markersize=10)
plt.axhline(y=mean_heart_rate, color='k', linestyle='--', label=f'Средна HR: {mean_heart_rate:.1f} BPM')
plt.title('Вариабилност на сърдечната честота', fontsize=16)
plt.xlabel('Време (s)', fontsize=14)
plt.ylabel('Моментна сърдечна честота (BPM)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Бонус: Симулиране на различни типове ЕКГ патологии

# 1. Функция за генериране на аритмични ЕКГ данни (нерегулярен сърдечен ритъм)
def generate_arrhythmic_ekg(duration=10, sampling_rate=500, base_hr=70, hr_variability=20):
    """
    Генерира ЕКГ с аритмии (нерегулярен ритъм)
    """
    t = np.arange(0, duration, 1/sampling_rate)
    ekg = np.zeros_like(t)
    baseline = 0.05 * np.sin(2 * np.pi * 0.25 * t) + 0.01 * np.random.randn(len(t))
    
    # Нерегулярни интервали между ударите
    beat_times = [0.5]
    while beat_times[-1] < duration:
        # Следващият удар идва с променлив интервал
        next_interval = 60 / (base_hr + np.random.uniform(-hr_variability, hr_variability))
        beat_times.append(beat_times[-1] + next_interval)
    
    # Генериране на PQRST вълни
    for beat_time in beat_times:
        if beat_time >= duration:
            break
            
        # P-вълна
        p_width = 0.08
        p_center = beat_time - 0.15
        p_wave = 0.15 * np.exp(-((t - p_center) ** 2) / (2 * p_width ** 2))
        
        # QRS комплекс
        q_center = beat_time - 0.05
        q_wave = -0.1 * np.exp(-((t - q_center) ** 2) / (2 * 0.01 ** 2))
        
        r_center = beat_time
        r_wave = 1.0 * np.exp(-((t - r_center) ** 2) / (2 * 0.01 ** 2))
        
        s_center = beat_time + 0.05
        s_wave = -0.2 * np.exp(-((t - s_center) ** 2) / (2 * 0.02 ** 2))
        
        # T-вълна
        t_width = 0.1
        t_center = beat_time + 0.25
        t_wave = 0.3 * np.exp(-((t - t_center) ** 2) / (2 * t_width ** 2))
        
        # Комбиниране на всички компоненти
        ekg += p_wave + q_wave + r_wave + s_wave + t_wave
    
    # Добавяне на базова линия и шум
    ekg += baseline + 0.03 * np.random.randn(len(t))
    
    return t, ekg

# 2. Функция за генериране на ЕКГ с пропуснати удари (синус аритмия)
def generate_skipped_beats_ekg(duration=10, sampling_rate=500, heart_rate=70, skip_probability=0.2):
    """
    Генерира ЕКГ с пропуснати удари (синус аритмия)
    """
    t = np.arange(0, duration, 1/sampling_rate)
    ekg = np.zeros_like(t)
    baseline = 0.05 * np.sin(2 * np.pi * 0.25 * t) + 0.01 * np.random.randn(len(t))
    
    # Интервал между ударите
    beat_interval = 60 / heart_rate
    
    for beat_time in np.arange(0.5, duration, beat_interval):
        # Случайно пропускане на удари
        if np.random.random() < skip_probability:
            continue
            
        # P-вълна
        p_width = 0.08
        p_center = beat_time - 0.15
        p_wave = 0.15 * np.exp(-((t - p_center) ** 2) / (2 * p_width ** 2))
        
        # QRS комплекс
        q_center = beat_time - 0.05
        q_wave = -0.1 * np.exp(-((t - q_center) ** 2) / (2 * 0.01 ** 2))
        
        r_center = beat_time
        r_wave = 1.0 * np.exp(-((t - r_center) ** 2) / (2 * 0.01 ** 2))
        
        s_center = beat_time + 0.05
        s_wave = -0.2 * np.exp(-((t - s_center) ** 2) / (2 * 0.02 ** 2))
        
        # T-вълна
        t_width = 0.1
        t_center = beat_time + 0.25
        t_wave = 0.3 * np.exp(-((t - t_center) ** 2) / (2 * t_width ** 2))
        
        # Комбиниране на всички компоненти
        ekg += p_wave + q_wave + r_wave + s_wave + t_wave
    
    # Добавяне на базова линия и шум
    ekg += baseline + 0.03 * np.random.randn(len(t))
    
    return t, ekg

# Генериране на патологични ЕКГ данни
t_normal, ekg_normal = generate_synthetic_ekg(duration=10, heart_rate=70)
t_arrhythmic, ekg_arrhythmic = generate_arrhythmic_ekg(duration=10, base_hr=70, hr_variability=30)
t_skipped, ekg_skipped = generate_skipped_beats_ekg(duration=10, heart_rate=70, skip_probability=0.3)

# Визуализация на различните типове ЕКГ
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

axs[0].plot(t_normal, ekg_normal, 'b-')
axs[0].set_title('Нормален ЕКГ', fontsize=14)
axs[0].set_ylabel('Амплитуда (mV)', fontsize=12)
axs[0].grid(True)

axs[1].plot(t_arrhythmic, ekg_arrhythmic, 'r-')
axs[1].set_title('ЕКГ с аритмия (вариабилен ритъм)', fontsize=14)
axs[1].set_ylabel('Амплитуда (mV)', fontsize=12)
axs[1].grid(True)

axs[2].plot(t_skipped, ekg_skipped, 'g-')
axs[2].set_title('ЕКГ с пропуснати удари', fontsize=14)
axs[2].set_xlabel('Време (s)', fontsize=12)
axs[2].set_ylabel('Амплитуда (mV)', fontsize=12)
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Анализ на всички три ЕКГ записа
heart_rates = []
for i, (t, ekg) in enumerate([(t_normal, ekg_normal), 
                              (t_arrhythmic, ekg_arrhythmic), 
                              (t_skipped, ekg_skipped)]):
    # Изчисляване на сърдечната честота
    mean_hr, instant_hr, r_peaks = calculate_heart_rate(t, ekg, threshold=0.5)
    heart_rates.append((mean_hr, np.std(instant_hr)))
    
    # Изчисляване на първа и втора производна
    first_deriv, second_deriv = compute_derivatives(t, ekg)
    
    # Визуализация на вариабилността на сърдечната честота
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, ekg, 'b-')
    plt.plot(t[r_peaks], ekg[r_peaks], 'ro')
    plt.title(f'ЕКГ тип {i+1}', fontsize=14)
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    if len(instant_hr) > 0:
        plt.plot(t[r_peaks[:-1]], instant_hr, 'r.-')
        plt.axhline(y=mean_hr, color='k', linestyle='--', 
                    label=f'Средна HR: {mean_hr:.1f} BPM, STD: {np.std(instant_hr):.1f}')
        plt.title('Вариабилност на сърдечната честота', fontsize=14)
        plt.xlabel('Време (s)', fontsize=12)
        plt.ylabel('BPM', fontsize=12)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Сравнение на показателите за вариабилност на сърдечната честота
types = ['Нормален ЕКГ', 'ЕКГ с аритмия', 'ЕКГ с пропуснати удари']
x = np.arange(len(types))
mean_hrs = [hr[0] for hr in heart_rates]
std_hrs = [hr[1] for hr in heart_rates]

plt.figure(figsize=(12, 6))
plt.bar(x, mean_hrs, yerr=std_hrs, capsize=10, color=['blue', 'red', 'green'])
plt.xticks(x, types, fontsize=12)
plt.ylabel('Средна сърдечна честота (BPM)', fontsize=14)
plt.title('Сравнение на сърдечната честота и вариабилността ѝ', fontsize=16)
plt.grid(True, axis='y')

# Добавяне на текстови анотации за стандартното отклонение
for i, (mean, std) in enumerate(zip(mean_hrs, std_hrs)):
    plt.text(i, mean + std + 2, f'STD: {std:.1f}', ha='center', fontsize=12)

plt.tight_layout()
plt.show()

# Заключения
print("Заключения от анализа на ЕКГ данни:")
print("1. Локалните екстремуми (максимуми и минимуми) съответстват на ключови точки в ЕКГ сигнала:")
print("   - R-пик: главен позитивен максимум")
print("   - Q и S точки: локални минимуми около R-пика")
print("   - P и T вълни: по-малки максимуми")
print()
print("2. Първата производна (df/dt) показва скоростта на промяна на сигнала:")
print("   - Стръмни участъци в ЕКГ (като възходящия край на R-пика) имат висока положителна производна")
print("   - Низходящите участъци имат отрицателна производна")
print("   - Нулата на първата производна съответства на локалните екстремуми на сигнала")
print()
print("3. Втората производна (d²f/dt²) разкрива изпъкналостта/вдлъбнатостта на сигнала:")
print("   - Отрицателната втора производна (изпъкнала част) се среща при върховете на вълните")
print("   - Положителната втора производна (вдлъбната част) се среща при долините на вълните")
print()
print("4. Анализът на вариабилността на сърдечната честота показва значителни разлики между трите типа ЕКГ:")
print(f"   - Нормален ЕКГ: Средна HR = {heart_rates[0][0]:.1f} BPM, STD = {heart_rates[0][1]:.1f}")
print(f"   - ЕКГ с аритмия: Средна HR = {heart_rates[1][0]:.1f} BPM, STD = {heart_rates[1][1]:.1f}")
print(f"   - ЕКГ с пропуснати удари: Средна HR = {heart_rates[2][0]:.1f} BPM, STD = {heart_rates[2][1]:.1f}")
