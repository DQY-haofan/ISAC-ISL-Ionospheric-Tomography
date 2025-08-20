#!/usr/bin/env python3
"""
ISAC Noise Model - High-Fidelity Instrument Phase Noise Simulator for ISL

This module implements a comprehensive phase noise model for inter-satellite links (ISL)
used in space weather monitoring through ionospheric TEC measurements.

Author: ISAC Project Team
Date: 2025
"""

import numpy as np
from scipy import signal, integrate, linalg, optimize
from typing import Tuple, Optional
import warnings


class IsacNoiseModel:
    """
    High-fidelity instrument phase noise model for inter-satellite Ka-band links.
    
    This model simulates the composite instrument phase noise φ_instr that corrupts
    ionospheric TEC measurements in satellite-to-satellite RF links.
    
    The total instrument phase noise consists of:
    - Local oscillator (LO) phase noise shaped by the PLL
    - Thermal drift residual noise shaped by the PLL  
    - ADC equivalent phase noise (white noise from jitter and quantization)
    """
    
    def __init__(
        self,
        oscillator_type: str,
        temperature_profile_K: np.ndarray,
        time_vector_s: np.ndarray,
        adc_enob: float,
        f_carrier_Hz: float,
        f_if_Hz: float,
        t_jitter_rms_s: float,
        pll_bandwidth_Hz: float
    ):
        """
        Initialize the ISAC noise model with hardware and environmental parameters.
        
        Parameters
        ----------
        oscillator_type : str
            Type of oscillator: 'Typical_OCXO' or 'High_Performance_USO'
        temperature_profile_K : np.ndarray
            Temperature profile over time in Kelvin
        time_vector_s : np.ndarray
            Time vector corresponding to temperature profile in seconds
        adc_enob : float
            ADC effective number of bits (ENOB)
        f_carrier_Hz : float
            ISL carrier frequency in Hz (e.g., 26e9 for Ka-band)
        f_if_Hz : float
            Intermediate frequency before ADC sampling in Hz (e.g., 500e6)
        t_jitter_rms_s : float
            RMS clock jitter in seconds (e.g., 100e-15)
        pll_bandwidth_Hz : float
            PLL equivalent noise bandwidth in Hz
        """
        # Store input parameters
        self.oscillator_type = oscillator_type
        self.temperature_profile_K = temperature_profile_K
        self.time_vector_s = time_vector_s
        self.adc_enob = adc_enob
        self.f_carrier_Hz = f_carrier_Hz
        self.f_if_Hz = f_if_Hz
        self.t_jitter_rms_s = t_jitter_rms_s
        self.pll_bandwidth_Hz = pll_bandwidth_Hz
        
        # Compute derived parameters
        self.dt = np.mean(np.diff(time_vector_s))
        self.fs = 1.0 / self.dt  # Sampling frequency
        self.n_samples = len(time_vector_s)
        
        # Initialize oscillator parameters
        self._initialize_oscillator_params()
        
    def _initialize_oscillator_params(self):
        """Initialize oscillator-specific h_alpha coefficients."""
        if self.oscillator_type == 'High_Performance_USO':
            self.h_alpha = self._fit_uso_coefficients()
            # Thermal drift coefficients (placeholder values, should be calibrated)
            self.h_thermal = {
                -2: 1e-24,  # Random Walk FM
                -1: 1e-22   # Flicker FM  
            }
        elif self.oscillator_type == 'Typical_OCXO':
            # Typical OCXO coefficients (example values)
            self.h_alpha = {
                -1: 1e-18,  # Flicker FM
                0: 1e-19,   # White FM
                1: 1e-20,   # Flicker PM
                2: 1e-21    # White PM
            }
            self.h_thermal = {
                -2: 1e-23,
                -1: 1e-21
            }
        else:
            raise ValueError(f"Unknown oscillator type: {self.oscillator_type}")
    
    def _fit_uso_coefficients(self) -> dict:
        """
        Fit h_alpha coefficients for High-Performance USO from SSB phase noise data.
        
        Returns
        -------
        dict
            Dictionary of h_alpha coefficients indexed by alpha value
        """
        # SSB phase noise data for 10 MHz USO
        f_offset_Hz = np.array([1, 10, 100, 1000])
        L_f_dBc = np.array([-80, -100, -117, -119])
        
        # Convert SSB to double-sideband PSD: S_phi(f) = 2 * 10^(L(f)/10)
        S_phi_f = 2 * 10**(L_f_dBc / 10)
        
        # Scale to carrier frequency (f_c^2 / f_ref^2)
        f_ref = 10e6  # Reference oscillator frequency
        scale_factor = (self.f_carrier_Hz / f_ref)**2
        S_phi_scaled = S_phi_f * scale_factor
        
        # Fit power law model: S_phi(f) = sum(h_alpha * f^alpha)
        # Using log-log linear regression for simplified fitting
        def power_law_model(f, h_vals):
            """Power law model for phase noise."""
            result = np.zeros_like(f)
            alphas = [-1, 0, 1, 2]
            for i, alpha in enumerate(alphas):
                if i < len(h_vals):
                    result += h_vals[i] * f**alpha
            return result
        
        # Simplified fitting using least squares in log-log space
        log_f = np.log10(f_offset_Hz)
        log_S = np.log10(S_phi_scaled)
        
        # Estimate dominant slope
        slope = np.polyfit(log_f[1:3], log_S[1:3], 1)[0]
        
        # Assign coefficients based on typical USO behavior
        h_alpha = {}
        if slope < -1.5:
            h_alpha[-1] = S_phi_scaled[1] * f_offset_Hz[1]  # Flicker FM dominant
            h_alpha[0] = S_phi_scaled[2] * 0.1  # Small white FM
        else:
            h_alpha[0] = S_phi_scaled[2]  # White FM dominant
            h_alpha[-1] = S_phi_scaled[1] * f_offset_Hz[1] * 0.1
        
        h_alpha[1] = S_phi_scaled[3] / f_offset_Hz[3] * 0.01  # Small Flicker PM
        h_alpha[2] = S_phi_scaled[3] / (f_offset_Hz[3]**2) * 0.001  # Small White PM
        
        return h_alpha
    
    def _calculate_lo_psd(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate local oscillator phase noise PSD.
        
        Parameters
        ----------
        f : np.ndarray
            Frequency vector in Hz
        
        Returns
        -------
        np.ndarray
            LO phase noise PSD in rad^2/Hz
        """
        S_phi_lo = np.zeros_like(f)
        
        # Avoid division by zero at f=0
        f_safe = np.where(f > 0, f, 1e-10)
        
        # Allan variance to phase noise conversion factor
        f_c_squared = self.f_carrier_Hz**2
        
        # Sum contributions from different noise processes
        for alpha, h_val in self.h_alpha.items():
            with np.errstate(divide='ignore', invalid='ignore'):
                contribution = h_val * f_safe**alpha
                contribution = np.where(np.isfinite(contribution), contribution, 0)
                S_phi_lo += contribution
        
        # Apply frequency scaling: S_phi,LO(f) = (f_c^2/f^2) * S_y(f)
        with np.errstate(divide='ignore', invalid='ignore'):
            S_phi_lo = (f_c_squared / f_safe**2) * S_phi_lo
            S_phi_lo = np.where(np.isfinite(S_phi_lo), S_phi_lo, 0)
        
        return S_phi_lo
    
    def _calculate_drift_psd(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate thermal drift residual phase noise PSD.
        
        Parameters
        ----------
        f : np.ndarray
            Frequency vector in Hz
        
        Returns
        -------
        np.ndarray
            Drift phase noise PSD in rad^2/Hz
        """
        S_phi_drift = np.zeros_like(f)
        
        # Avoid division by zero
        f_safe = np.where(f > 0, f, 1e-10)
        
        # Frequency scaling factor
        f_c_squared = self.f_carrier_Hz**2
        
        # Thermal drift contributions (low-frequency noise)
        for alpha, h_val in self.h_thermal.items():
            with np.errstate(divide='ignore', invalid='ignore'):
                contribution = h_val * f_safe**alpha
                contribution = np.where(np.isfinite(contribution), contribution, 0)
                S_phi_drift += contribution
        
        # Apply frequency scaling
        with np.errstate(divide='ignore', invalid='ignore'):
            S_phi_drift = (f_c_squared / f_safe**2) * S_phi_drift
            S_phi_drift = np.where(np.isfinite(S_phi_drift), S_phi_drift, 0)
        
        return S_phi_drift
    
    def _calculate_adc_psd(self) -> float:
        """
        Calculate ADC equivalent phase noise PSD (white noise).
        
        Returns
        -------
        float
            ADC phase noise PSD in rad^2/Hz (constant across frequency)
        """
        # Jitter-dominated SNR
        snr_j_dB = -20 * np.log10(2 * np.pi * self.f_if_Hz * self.t_jitter_rms_s)
        rho_j = 10**(snr_j_dB / 10)
        
        # Quantization-dominated SNR
        snr_q_dB = 6.02 * self.adc_enob + 1.76
        rho_q = 10**(snr_q_dB / 10)
        
        # Total ADC SNR (minimum of the two)
        rho_adc = min(rho_j, rho_q)
        
        # Single-sample phase noise variance
        var_phi_adc = 1 / (2 * rho_adc)
        
        # Convert to single-sideband PSD
        # Factor of 2 converts from double-sideband to single-sideband
        S_phi_adc = (var_phi_adc / self.fs) * 2
        
        return S_phi_adc
    
    def _pll_transfer_function(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate PLL transfer function |H_PLL(f)|^2.
        
        Parameters
        ----------
        f : np.ndarray
            Frequency vector in Hz
        
        Returns
        -------
        np.ndarray
            PLL transfer function squared (dimensionless)
        """
        # First-order low-pass filter model
        f_c = self.pll_bandwidth_Hz
        H_pll_squared = 1 / (1 + (f / f_c)**2)
        return H_pll_squared
    
    def get_instrument_noise_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate total instrument phase noise power spectral density.
        
        Returns
        -------
        tuple
            (frequency_Hz, psd_rad2_per_Hz) where:
            - frequency_Hz: Frequency vector from 0 to fs/2
            - psd_rad2_per_Hz: Single-sideband PSD in rad^2/Hz
        """
        # Create frequency vector (positive frequencies only)
        frequency_Hz = np.fft.rfftfreq(self.n_samples, self.dt)
        
        # Calculate individual PSD components
        S_phi_lo = self._calculate_lo_psd(frequency_Hz)
        S_phi_drift = self._calculate_drift_psd(frequency_Hz)
        S_phi_adc = self._calculate_adc_psd()
        
        # Apply PLL shaping to LO and drift noise
        H_pll_squared = self._pll_transfer_function(frequency_Hz)
        
        # Total PSD: S_phi^(total)(f) = |H_PLL(f)|^2 * [S_phi,LO(f) + S_phi,drift(f)] + S_phi,ADC
        psd_rad2_per_Hz = H_pll_squared * (S_phi_lo + S_phi_drift) + S_phi_adc
        
        # Ensure no negative or NaN values
        psd_rad2_per_Hz = np.where(psd_rad2_per_Hz > 0, psd_rad2_per_Hz, 1e-30)
        
        return frequency_Hz, psd_rad2_per_Hz
    
    def generate_instrument_noise_timeseries(self) -> np.ndarray:
        """
        Generate instrument phase noise time series using frequency domain synthesis.
        
        Returns
        -------
        np.ndarray
            Phase noise time series in radians, same length as time_vector_s
        """
        # Get PSD
        freq_Hz, psd_rad2_per_Hz = self.get_instrument_noise_psd()
        
        # Convert PSD to amplitude spectrum for frequency domain synthesis
        # For real signals, we need to account for both positive and negative frequencies
        # Single-sideband PSD to amplitude: sqrt(PSD * df * 2) for f>0, sqrt(PSD * df) for f=0
        df = freq_Hz[1] - freq_Hz[0]
        
        # Generate random phases
        random_phases = np.random.uniform(-np.pi, np.pi, len(freq_Hz))
        
        # Create complex spectrum
        amplitude = np.sqrt(psd_rad2_per_Hz * df)
        amplitude[1:] *= np.sqrt(2)  # Account for negative frequencies (except DC)
        
        # Complex spectrum with random phases
        spectrum = amplitude * np.exp(1j * random_phases)
        spectrum[0] = np.real(spectrum[0])  # DC component must be real
        
        # IFFT to get time series
        noise_timeseries = np.fft.irfft(spectrum, n=self.n_samples)
        
        return noise_timeseries
    
    def get_noise_covariance_matrix(self, num_samples: int) -> np.ndarray:
        """
        Calculate noise covariance matrix in time domain (Toeplitz structure).
        
        Parameters
        ----------
        num_samples : int
            Number of samples for the covariance matrix
        
        Returns
        -------
        np.ndarray
            Toeplitz covariance matrix of size (num_samples x num_samples)
        """
        if num_samples > self.n_samples:
            warnings.warn(f"Requested {num_samples} samples exceeds available {self.n_samples}")
            num_samples = min(num_samples, self.n_samples)
        
        # Get PSD
        freq_Hz, psd_rad2_per_Hz = self.get_instrument_noise_psd()
        
        # Calculate autocorrelation function via inverse Fourier transform
        # C(Δt) = ∫ S_phi(f) * cos(2πfΔt) df
        
        # Time lags for autocorrelation
        max_lag = num_samples - 1
        lags = np.arange(max_lag + 1) * self.dt
        
        # Calculate autocorrelation for each lag
        autocorr = np.zeros(len(lags))
        
        for i, lag in enumerate(lags):
            # Integrate PSD * cos(2πfΔt) over frequency
            integrand = psd_rad2_per_Hz * np.cos(2 * np.pi * freq_Hz * lag)
            autocorr[i] = np.trapz(integrand, freq_Hz)
        
        # Build Toeplitz covariance matrix
        # [Σ_n]_ij = C(|i-j| * T_s)
        covariance_matrix = linalg.toeplitz(autocorr[:num_samples])
        
        # Ensure positive semi-definite (numerical stability)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        covariance_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return covariance_matrix


if __name__ == "__main__":
    """
    Example usage of the IsacNoiseModel class with visualization.
    """
    import matplotlib.pyplot as plt
    
    # Set up time vector (10 seconds at 10 Hz sampling)
    duration_s = 10.0
    fs = 10.0  # Hz
    time_vector_s = np.arange(0, duration_s, 1/fs)
    n_samples = len(time_vector_s)
    
    # Simulate temperature profile (sinusoidal variation around 290K)
    temperature_mean_K = 290.0
    temperature_amplitude_K = 5.0
    temperature_period_s = 100.0
    temperature_profile_K = (temperature_mean_K + 
                            temperature_amplitude_K * 
                            np.sin(2 * np.pi * time_vector_s / temperature_period_s))
    
    # Initialize the noise model with typical parameters
    model = IsacNoiseModel(
        oscillator_type='High_Performance_USO',
        temperature_profile_K=temperature_profile_K,
        time_vector_s=time_vector_s,
        adc_enob=12.0,  # 12-bit ENOB
        f_carrier_Hz=26e9,  # 26 GHz Ka-band
        f_if_Hz=500e6,  # 500 MHz IF
        t_jitter_rms_s=100e-15,  # 100 fs RMS jitter
        pll_bandwidth_Hz=100.0  # 100 Hz PLL bandwidth
    )
    
    # Get instrument noise PSD
    freq_Hz, psd_rad2_per_Hz = model.get_instrument_noise_psd()
    
    # Generate noise time series
    noise_timeseries = model.generate_instrument_noise_timeseries()
    
    # Get covariance matrix (for first 50 samples)
    num_cov_samples = min(50, n_samples)
    covariance_matrix = model.get_noise_covariance_matrix(num_cov_samples)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ISAC Noise Model - Instrument Phase Noise Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: PSD (log-log)
    ax1 = axes[0, 0]
    valid_freq = freq_Hz[1:]  # Skip DC
    valid_psd = psd_rad2_per_Hz[1:]
    ax1.loglog(valid_freq, valid_psd, 'b-', linewidth=1.5, label='Total PSD')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Phase Noise PSD [rad²/Hz]')
    ax1.set_title('Power Spectral Density')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()
    ax1.set_xlim([1e-2, fs/2])
    
    # Plot 2: Time series (first 2 seconds)
    ax2 = axes[0, 1]
    plot_duration = min(2.0, duration_s)
    plot_samples = int(plot_duration * fs)
    ax2.plot(time_vector_s[:plot_samples], noise_timeseries[:plot_samples], 
             'g-', linewidth=0.8, alpha=0.8)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Phase Noise [rad]')
    ax2.set_title(f'Noise Time Series (first {plot_duration}s)')
    ax2.grid(True, alpha=0.3)
    
    # Add RMS value annotation
    rms_value = np.std(noise_timeseries)
    ax2.text(0.02, 0.98, f'RMS: {rms_value:.3e} rad', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Covariance matrix heatmap
    ax3 = axes[1, 0]
    im = ax3.imshow(covariance_matrix, cmap='RdBu_r', aspect='auto')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Sample Index')
    ax3.set_title(f'Covariance Matrix ({num_cov_samples}×{num_cov_samples})')
    plt.colorbar(im, ax=ax3, label='Covariance [rad²]')
    
    # Plot 4: Autocorrelation function
    ax4 = axes[1, 1]
    autocorr_lags = np.arange(num_cov_samples) * (1/fs)
    autocorr_values = covariance_matrix[0, :]
    ax4.plot(autocorr_lags, autocorr_values, 'r-', linewidth=1.5)
    ax4.set_xlabel('Time Lag [s]')
    ax4.set_ylabel('Autocorrelation [rad²]')
    ax4.set_title('Autocorrelation Function')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ISAC NOISE MODEL SUMMARY")
    print("="*60)
    print(f"Oscillator Type: {model.oscillator_type}")
    print(f"Carrier Frequency: {model.f_carrier_Hz/1e9:.1f} GHz")
    print(f"IF Frequency: {model.f_if_Hz/1e6:.1f} MHz")
    print(f"ADC ENOB: {model.adc_enob:.1f} bits")
    print(f"Clock Jitter: {model.t_jitter_rms_s*1e15:.1f} fs RMS")
    print(f"PLL Bandwidth: {model.pll_bandwidth_Hz:.1f} Hz")
    print(f"\nNoise Statistics:")
    print(f"  Phase Noise RMS: {rms_value:.3e} rad")
    print(f"  Phase Noise RMS: {rms_value*180/np.pi:.3e} deg")
    print(f"  Allan Deviation @ 1s: {rms_value/np.sqrt(fs):.3e}")
    print("="*60)