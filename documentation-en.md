# ECG Analysis Using Calculus

## Summary

This project demonstrates the application of calculus for analyzing electrocardiographic (ECG) data. The project includes generating synthetic ECG data, extracting key features through differentiation, analyzing local extrema, examining the convexity/concavity of the signal, and calculating heart rate.

## Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Project Structure](#project-structure)
4. [Theoretical Foundation](#theoretical-foundation)
5. [Core Functions](#core-functions)
6. [Usage Examples](#usage-examples)
7. [Extensions and Pathology Simulations](#extensions-and-pathology-simulations)
8. [Conclusions](#conclusions)
9. [References](#references)

## Introduction

Electrocardiography (ECG) is a method for recording the electrical activity of the heart over time. The ECG signal represents a one-dimensional function of time, making it perfectly suited for analysis using calculus methods. In this project, we apply concepts such as derivatives, local extrema, and curvature analysis to study ECG signals.

## Installation and Setup

To use this code, you will need the following libraries:

```bash
pip install numpy matplotlib scipy
```

## Project Structure

The main Jupyter Notebook contains the following components:

1. Generation of synthetic ECG data
2. Analysis of local extrema
3. Calculation and analysis of derivatives
4. Heart rate assessment
5. Simulation of pathological ECG recordings

## Theoretical Foundation

### ECG Components

A normal ECG signal consists of the following key components:

- **P-wave**: Represents atrial depolarization
- **QRS complex**: Represents ventricular depolarization
  - **Q-wave**: Initial negative deflection
  - **R-peak**: Main positive peak
  - **S-wave**: Negative deflection after the R-peak
- **T-wave**: Represents ventricular repolarization

### Application of Calculus

In this project, we apply the following calculus concepts:

1. **Local Extrema**: Points where the first derivative equals zero
   - **Local Maxima**: Points where the first derivative equals zero and the second derivative is negative
   - **Local Minima**: Points where the first derivative equals zero and the second derivative is positive

2. **Derivatives**:
   - **First Derivative**: Rate of change of the ECG signal
   - **Second Derivative**: Curvature/convexity of the signal

## Core Functions

### Generating ECG Data

```python
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
```

### Finding Local Extrema

```python
def find_extrema(signal, threshold=0.1):
    """
    Finds local maxima and minima of the signal
    
    Parameters:
    - signal: ECG signal
    - threshold: threshold for peak detection
    
    Returns:
    - locations of maxima
    - locations of minima
    """
```

### Calculating Derivatives

```python
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
```

### Analyzing Convexity/Concavity

```python
def analyze_concavity(second_derivative):
    """
    Analyzes convexity/concavity of the graph based on the second derivative
    
    Parameters:
    - second_derivative: second derivative of the signal
    
    Returns:
    - indices of convex parts (second derivative < 0)
    - indices of concave parts (second derivative > 0)
    """
```

### Measuring Heart Rate

```python
def calculate_heart_rate(t, ekg_signal, threshold=0.5):
    """
    Calculates heart rate from the R-peaks of the ECG signal
    
    Parameters:
    - t: time axis
    - ekg_signal: ECG signal
    - threshold: threshold for R-peak detection
    
    Returns:
    - mean heart rate in beats per minute (BPM)
    - instantaneous heart rate from each R-R interval
    """
```

## Usage Examples

```python
# Generate synthetic ECG data
t, ekg_signal = generate_synthetic_ekg(duration=5, heart_rate=70)

# Find local extrema
maxima_indices, minima_indices = find_extrema(ekg_signal, threshold=0.1)

# Calculate derivatives
first_deriv, second_deriv = compute_derivatives(t, ekg_signal)

# Analyze convexity/concavity
convex_indices, concave_indices = analyze_concavity(second_deriv)

# Calculate heart rate
mean_heart_rate, instant_heart_rate, r_peaks = calculate_heart_rate(t, ekg_signal, threshold=0.5)
```

## Extensions and Pathology Simulations

The project includes generation and analysis of pathological ECG recordings:

1. **ECG with Arrhythmia (Irregular Rhythm)**:
   ```python
   t_arrhythmic, ekg_arrhythmic = generate_arrhythmic_ekg(duration=10, base_hr=70, hr_variability=30)
   ```

2. **ECG with Skipped Beats**:
   ```python
   t_skipped, ekg_skipped = generate_skipped_beats_ekg(duration=10, heart_rate=70, skip_probability=0.3)
   ```

## Conclusions

1. Local extrema (maxima and minima) correspond to key points in the ECG signal:
   - R-peak: main positive maximum
   - Q and S points: local minima around the R-peak
   - P and T waves: smaller maxima

2. The first derivative (df/dt) shows the rate of change of the signal:
   - Steep sections in the ECG have a high positive derivative
   - Descending sections have a negative derivative
   - Zero first derivative corresponds to local extrema

3. The second derivative (d²f/dt²) reveals the convexity/concavity of the signal:
   - Negative second derivative: convex part (wave peaks)
   - Positive second derivative: concave part (wave valleys)

4. Heart rate variability analysis shows significant differences between different types of ECG recordings.

## References

1. Malmivuo, J., & Plonsey, R. (1995). *Bioelectromagnetism: Principles and Applications of Bioelectric and Biomagnetic Fields*. Oxford University Press.
2. Goldberger, A. L., Amaral, L. A., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220.
3. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, (3), 230-236.
