import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy import signal

# Creating synthetic ECG data
def generate_synthetic_ekg(duration=10, sampling_rate=500, heart_rate=70):
    """
    Generates synthetic ECG data
    
    Parameters:
    - duration: duration of the ECG recording in seconds
    - sampling_rate: sampling frequency in Hz
    - heart_rate: heart rate in beats per minute
    
    Returns:
    - time axis
    - ECG signal
    """
    # Calculate beat interval in seconds
    beat_interval = 60 / heart_rate  
    
    # Create time axis
    t = np.arange(0, duration, 1/sampling_rate)
    
    # Baseline with small noise
    baseline = 0.05 * np.sin(2 * np.pi * 0.25 * t) + 0.01 * np.random.randn(len(t))
    
    # Initialize ECG signal
    ekg = np.zeros_like(t)
    
    # Add PQRST components to each heartbeat
    for beat_time in np.arange(0.5, duration, beat_interval):
        # P-wave
        p_width = 0.08
        p_center = beat_time - 0.15
        p_wave = 0.15 * np.exp(-((t - p_center) ** 2) / (2 * p_width ** 2))
        
        # QRS complex
        q_center = beat_time - 0.05
        q_wave = -0.1 * np.exp(-((t - q_center) ** 2) / (2 * 0.01 ** 2))
        
        r_center = beat_time
        r_wave = 1.0 * np.exp(-((t - r_center) ** 2) / (2 * 0.01 ** 2))
        
        s_center = beat_time + 0.05
        s_wave = -0.2 * np.exp(-((t - s_center) ** 2) / (2 * 0.02 ** 2))
        
        # T-wave
        t_width = 0.1
        t_center = beat_time + 0.25
        t_wave = 0.3 * np.exp(-((t - t_center) ** 2) / (2 * t_width ** 2))
        
        # Combine all components
        ekg += p_wave + q_wave + r_wave + s_wave + t_wave
    
    # Add baseline and random noise
    ekg += baseline + 0.02 * np.random.randn(len(t))
    
    return t, ekg

# Function to find local extrema (maxima and minima)
def find_extrema(signal, threshold=0.1):
    """
    Finds the local maxima and minima of the signal
    
    Parameters:
    - signal: ECG signal
    - threshold: threshold for peak detection
    
    Returns:
    - locations of maxima
    - locations of minima
    """
    # Find local maxima
    maxima, _ = find_peaks(signal, height=threshold)
    
    # Find local minima by inverting the signal
    minima, _ = find_peaks(-signal, height=threshold)
    
    return maxima, minima

# Function to calculate first and second derivatives
def compute_derivatives(t, signal):
    """
    Calculates the first and second derivatives of the ECG signal
    
    Parameters:
    - t: time axis
    - signal: ECG signal
    
    Returns:
    - first derivative
    - second derivative
    """
    # Calculate the first derivative: df/dt
    first_derivative = np.gradient(signal, t)
    
    # Calculate the second derivative: d²f/dt²
    second_derivative = np.gradient(first_derivative, t)
    
    return first_derivative, second_derivative

# Function to analyze convexity/concavity
def analyze_concavity(second_derivative):
    """
    Analyzes convexity/concavity of the graph based on the second derivative
    
    Parameters:
    - second_derivative: second derivative of the signal
    
    Returns:
    - indices of convex parts (second derivative < 0)
    - indices of concave parts (second derivative > 0)
    """
    # Convex part (second derivative < 0)
    convex_regions = np.where(second_derivative < 0)[0]
    
    # Concave part (second derivative > 0)
    concave_regions = np.where(second_derivative > 0)[0]
    
    return convex_regions, concave_regions

# Create synthetic ECG data
t, ekg_signal = generate_synthetic_ekg(duration=5, heart_rate=70)

# Find local extrema
maxima_indices, minima_indices = find_extrema(ekg_signal, threshold=0.1)

# Calculate derivatives
first_deriv, second_deriv = compute_derivatives(t, ekg_signal)

# Analyze convexity/concavity
convex_indices, concave_indices = analyze_concavity(second_deriv)

# Visualization of ECG data and key components
fig, axs = plt.subplots(3, 1, figsize=(15, 12))

# Plot of ECG signal with marked extrema
axs[0].plot(t, ekg_signal, 'b-', label='ECG signal')
axs[0].plot(t[maxima_indices], ekg_signal[maxima_indices], 'ro', label='Local maxima')
axs[0].plot(t[minima_indices], ekg_signal[minima_indices], 'go', label='Local minima')

# Add annotations for PQRST waves
# Find R-peaks (highest points)
r_peaks = maxima_indices[np.argsort(ekg_signal[maxima_indices])[-5:]]
for i, peak in enumerate(r_peaks[:3]):  # Only for the first three for clarity
    # Find Q and S points around the R-peak
    window_size = 50  # number of points to search around
    left_window = max(0, peak - window_size)
    right_window = min(len(ekg_signal), peak + window_size)
    
    q_idx = left_window + np.argmin(ekg_signal[left_window:peak])
    s_idx = peak + np.argmin(ekg_signal[peak:right_window])
    
    # Try to detect P and T waves
    p_window = max(0, q_idx - window_size)
    t_window = min(len(ekg_signal), s_idx + window_size)
    
    p_candidates = maxima_indices[(maxima_indices > p_window) & (maxima_indices < q_idx)]
    p_idx = p_candidates[0] if len(p_candidates) > 0 else None
    
    t_candidates = maxima_indices[(maxima_indices > s_idx) & (maxima_indices < t_window)]
    t_idx = t_candidates[0] if len(t_candidates) > 0 else None
    
    # Add annotations
    if p_idx:
        axs[0].annotate('P', (t[p_idx], ekg_signal[p_idx]), fontsize=12)
    axs[0].annotate('Q', (t[q_idx], ekg_signal[q_idx]), fontsize=12)
    axs[0].annotate('R', (t[peak], ekg_signal[peak]), fontsize=12)
    axs[0].annotate('S', (t[s_idx], ekg_signal[s_idx]), fontsize=12)
    if t_idx:
        axs[0].annotate('T', (t[t_idx], ekg_signal[t_idx]), fontsize=12)

axs[0].set_title('ECG signal with key points marked', fontsize=14)
axs[0].set_xlabel('Time (s)', fontsize=12)
axs[0].set_ylabel('Amplitude (mV)', fontsize=12)
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Plot of the first derivative
axs[1].plot(t, first_deriv, 'r-', label='First derivative')
axs[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[1].set_title('First derivative of the ECG signal', fontsize=14)
axs[1].set_xlabel('Time (s)', fontsize=12)
axs[1].set_ylabel('dV/dt', fontsize=12)
axs[1].legend(loc='upper right')
axs[1].grid(True)

# Plot of the second derivative with convexity/concavity analysis
axs[2].plot(t, second_deriv, 'g-', label='Second derivative')
axs[2].fill_between(t, second_deriv, 0, where=(second_deriv < 0), color='red', alpha=0.3, label='Convex part (d²V/dt² < 0)')
axs[2].fill_between(t, second_deriv, 0, where=(second_deriv > 0), color='blue', alpha=0.3, label='Concave part (d²V/dt² > 0)')
axs[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[2].set_title('Second derivative and convexity/concavity analysis', fontsize=14)
axs[2].set_xlabel('Time (s)', fontsize=12)
axs[2].set_ylabel('d²V/dt²', fontsize=12)
axs[2].legend(loc='upper right')
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Additional analysis: Measuring heart rate from R-R intervals
def calculate_heart_rate(t, ekg_signal, threshold=0.5):
    """
    Calculates heart rate from the R-peaks of the ECG signal
    
    Parameters:
    - t: time axis
    - ekg_signal: ECG signal
    - threshold: threshold for R-peak detection
    
    Returns:
    - average heart rate in beats per minute (BPM)
    - instantaneous heart rate from each R-R interval
    - R-peak indices
    """
    # Find R-peaks
    r_peaks, _ = find_peaks(ekg_signal, height=threshold)
    
    # Calculate R-R intervals in time units
    rr_intervals = np.diff(t[r_peaks])
    
    # Calculate instantaneous heart rate for each R-R interval (60 for conversion to minutes)
    instant_hr = 60 / rr_intervals
    
    # Calculate average heart rate
    mean_hr = np.mean(instant_hr)
    
    return mean_hr, instant_hr, r_peaks

# Calculate heart rate
mean_heart_rate, instant_heart_rate, r_peaks = calculate_heart_rate(t, ekg_signal, threshold=0.5)

# Visualization of heart rate
plt.figure(figsize=(12, 6))
plt.plot(t, ekg_signal, 'b-', label='ECG signal')
plt.plot(t[r_peaks], ekg_signal[r_peaks], 'ro', markersize=8, label='R-peaks')

# Add text with average heart rate
plt.text(0.5, max(ekg_signal) * 0.8, 
         f'Average heart rate: {mean_heart_rate:.1f} BPM', 
         fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

plt.title('Heart rate analysis from R-R intervals', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude (mV)', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Visualization of heart rate variability
plt.figure(figsize=(12, 6))
plt.plot(t[r_peaks[:-1]], instant_heart_rate, 'r.-', markersize=10)
plt.axhline(y=mean_heart_rate, color='k', linestyle='--', label=f'Average HR: {mean_heart_rate:.1f} BPM')
plt.title('Heart Rate Variability', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Instantaneous heart rate (BPM)', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Bonus: Simulating different types of ECG pathologies

# 1. Function to generate arrhythmic ECG data (irregular heart rhythm)
def generate_arrhythmic_ekg(duration=10, sampling_rate=500, base_hr=70, hr_variability=20):
    """
    Generates ECG with arrhythmias (irregular rhythm)
    """
    t = np.arange(0, duration, 1/sampling_rate)
    ekg = np.zeros_like(t)
    baseline = 0.05 * np.sin(2 * np.pi * 0.25 * t) + 0.01 * np.random.randn(len(t))
    
    # Irregular intervals between beats
    beat_times = [0.5]
    while beat_times[-1] < duration:
        # Next beat comes with a variable interval
        next_interval = 60 / (base_hr + np.random.uniform(-hr_variability, hr_variability))
        beat_times.append(beat_times[-1] + next_interval)
    
    # Generate PQRST waves
    for beat_time in beat_times:
        if beat_time >= duration:
            break
            
        # P-wave
        p_width = 0.08
        p_center = beat_time - 0.15
        p_wave = 0.15 * np.exp(-((t - p_center) ** 2) / (2 * p_width ** 2))
        
        # QRS complex
        q_center = beat_time - 0.05
        q_wave = -0.1 * np.exp(-((t - q_center) ** 2) / (2 * 0.01 ** 2))
        
        r_center = beat_time
        r_wave = 1.0 * np.exp(-((t - r_center) ** 2) / (2 * 0.01 ** 2))
        
        s_center = beat_time + 0.05
        s_wave = -0.2 * np.exp(-((t - s_center) ** 2) / (2 * 0.02 ** 2))
        
        # T-wave
        t_width = 0.1
        t_center = beat_time + 0.25
        t_wave = 0.3 * np.exp(-((t - t_center) ** 2) / (2 * t_width ** 2))
        
        # Combine all components
        ekg += p_wave + q_wave + r_wave + s_wave + t_wave
    
    # Add baseline and noise
    ekg += baseline + 0.03 * np.random.randn(len(t))
    
    return t, ekg

# 2. Function to generate ECG with skipped beats (sinus arrhythmia)
def generate_skipped_beats_ekg(duration=10, sampling_rate=500, heart_rate=70, skip_probability=0.2):
    """
    Generates ECG with skipped beats (sinus arrhythmia)
    """
    t = np.arange(0, duration, 1/sampling_rate)
    ekg = np.zeros_like(t)
    baseline = 0.05 * np.sin(2 * np.pi * 0.25 * t) + 0.01 * np.random.randn(len(t))
    
    # Interval between beats
    beat_interval = 60 / heart_rate
    
    for beat_time in np.arange(0.5, duration, beat_interval):
        # Randomly skip beats
        if np.random.random() < skip_probability:
            continue
            
        # P-wave
        p_width = 0.08
        p_center = beat_time - 0.15
        p_wave = 0.15 * np.exp(-((t - p_center) ** 2) / (2 * p_width ** 2))
        
        # QRS complex
        q_center = beat_time - 0.05
        q_wave = -0.1 * np.exp(-((t - q_center) ** 2) / (2 * 0.01 ** 2))
        
        r_center = beat_time
        r_wave = 1.0 * np.exp(-((t - r_center) ** 2) / (2 * 0.01 ** 2))
        
        s_center = beat_time + 0.05
        s_wave = -0.2 * np.exp(-((t - s_center) ** 2) / (2 * 0.02 ** 2))
        
        # T-wave
        t_width = 0.1
        t_center = beat_time + 0.25
        t_wave = 0.3 * np.exp(-((t - t_center) ** 2) / (2 * t_width ** 2))
        
        # Combine all components
        ekg += p_wave + q_wave + r_wave + s_wave + t_wave
    
    # Add baseline and noise
    ekg += baseline + 0.03 * np.random.randn(len(t))
    
    return t, ekg

# Generate pathological ECG data
t_normal, ekg_normal = generate_synthetic_ekg(duration=10, heart_rate=70)
t_arrhythmic, ekg_arrhythmic = generate_arrhythmic_ekg(duration=10, base_hr=70, hr_variability=30)
t_skipped, ekg_skipped = generate_skipped_beats_ekg(duration=10, heart_rate=70, skip_probability=0.3)

# Visualization of different ECG types
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

axs[0].plot(t_normal, ekg_normal, 'b-')
axs[0].set_title('Normal ECG', fontsize=14)
axs[0].set_ylabel('Amplitude (mV)', fontsize=12)
axs[0].grid(True)

axs[1].plot(t_arrhythmic, ekg_arrhythmic, 'r-')
axs[1].set_title('ECG with arrhythmia (variable rhythm)', fontsize=14)
axs[1].set_ylabel('Amplitude (mV)', fontsize=12)
axs[1].grid(True)

axs[2].plot(t_skipped, ekg_skipped, 'g-')
axs[2].set_title('ECG with skipped beats', fontsize=14)
axs[2].set_xlabel('Time (s)', fontsize=12)
axs[2].set_ylabel('Amplitude (mV)', fontsize=12)
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Analysis of all three ECG recordings
heart_rates = []
for i, (t, ekg) in enumerate([(t_normal, ekg_normal), 
                              (t_arrhythmic, ekg_arrhythmic), 
                              (t_skipped, ekg_skipped)]):
    # Calculate heart rate
    mean_hr, instant_hr, r_peaks = calculate_heart_rate(t, ekg, threshold=0.5)
    heart_rates.append((mean_hr, np.std(instant_hr)))
    
    # Calculate first and second derivatives
    first_deriv, second_deriv = compute_derivatives(t, ekg)
    
    # Visualization of heart rate variability
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, ekg, 'b-')
    plt.plot(t[r_peaks], ekg[r_peaks], 'ro')
    plt.title(f'ECG Type {i+1}', fontsize=14)
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    if len(instant_hr) > 0:
        plt.plot(t[r_peaks[:-1]], instant_hr, 'r.-')
        plt.axhline(y=mean_hr, color='k', linestyle='--', 
                    label=f'Average HR: {mean_hr:.1f} BPM, STD: {np.std(instant_hr):.1f}')
        plt.title('Heart Rate Variability', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('BPM', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Comparison of heart rate variability metrics
types = ['Normal ECG', 'ECG with arrhythmia', 'ECG with skipped beats']
x = np.arange(len(types))
mean_hrs = [hr[0] for hr in heart_rates]
std_hrs = [hr[1] for hr in heart_rates]

plt.figure(figsize=(12, 6))
plt.bar(x, mean_hrs, yerr=std_hrs, capsize=10, color=['blue', 'red', 'green'])
plt.xticks(x, types, fontsize=12)
plt.ylabel('Average heart rate (BPM)', fontsize=14)
plt.title('Comparison of heart rate and its variability', fontsize=16)
plt.grid(True, axis='y')

# Add text annotations for standard deviation
for i, (mean, std) in enumerate(zip(mean_hrs, std_hrs)):
    plt.text(i, mean + std + 2, f'STD: {std:.1f}', ha='center', fontsize=12)

plt.tight_layout()
plt.show()

# Conclusions
print("Conclusions from ECG data analysis:")
print("1. Local extrema (maxima and minima) correspond to key points in the ECG signal:")
print("   - R-peak: main positive maximum")
print("   - Q and S points: local minima around the R-peak")
print("   - P and T waves: smaller maxima")
print()
print("2. The first derivative (df/dt) shows the rate of change of the signal:")
print("   - Steep sections in the ECG (such as the ascending edge of the R-peak) have a high positive derivative")
print("   - Descending sections have a negative derivative")
print("   - Zero of the first derivative corresponds to the local extrema of the signal")
print()
print("3. The second derivative (d²f/dt²) reveals the convexity/concavity of the signal:")
print("   - Negative second derivative (convex part) occurs at the peaks of the waves")
print("   - Positive second derivative (concave part) occurs at the valleys of the waves")
print()
print("4. Heart rate variability analysis shows significant differences between the three types of ECG:")
print(f"   - Normal ECG: Average HR = {heart_rates[0][0]:.1f} BPM, STD = {heart_rates[0][1]:.1f}")
print(f"   - ECG with arrhythmia: Average HR = {heart_rates[1][0]:.1f} BPM, STD = {heart_rates[1][1]:.1f}")
print(f"   - ECG with skipped beats: Average HR = {heart_rates[2][0]:.1f} BPM, STD = {heart_rates[2][1]:.1f}")
