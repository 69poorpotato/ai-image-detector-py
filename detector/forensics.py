"""
High-performance forensic and signal-processing analyzer.
Vectorized for sub-100ms execution across multiple signal domains:
2D FFT radial power spectrum, noise residual kurtosis, local texture variance,
chromatic dispersion, histogram smoothness, and ELA recompression.
"""

from __future__ import annotations
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import scipy.stats
from PIL import Image

COMMON_AI_RESOLUTIONS = {
    (512, 512): "SD 1.5 standard square",
    (768, 768): "SD 2.1 standard square",
    (1024, 1024): "SDXL / Midjourney / DALL-E 3 square",
    (1152, 896): "SDXL landscape",
    (896, 1152): "SDXL portrait",
    (1216, 832): "SDXL widescreen",
    (832, 1216): "SDXL tall",
    (1344, 768): "SDXL cinematic 16:9",
    (768, 1344): "SDXL mobile 9:16",
    (1536, 640): "SDXL ultra-wide 2.4:1",
    (640, 1536): "SDXL ultra-tall",
    (1024, 1792): "DALL-E 3 portrait",
    (1792, 1024): "DALL-E 3 landscape",
    (1024, 1536): "Midjourney v6 portrait",
    (1536, 1024): "Midjourney v6 landscape",
    (1456, 816): "Flux.1 16:9 widescreen",
    (816, 1456): "Flux.1 9:16 portrait",
}


@dataclass
class ForensicResult:
    # Dimensions & aspect
    width: int = 0
    height: int = 0
    aspect_ratio: str = ""
    is_common_ai_resolution: bool = False
    ai_resolution_desc: Optional[str] = None

    # Frequency Domain (FFT)
    psd_slope: float = 0.0
    psd_r2_fit: float = 0.0
    spectral_grid_score: float = 0.0
    high_freq_energy_ratio: float = 0.0

    # Noise Residual & Statistics
    noise_variance: float = 0.0
    noise_kurtosis: float = 0.0
    noise_skewness: float = 0.0

    # Texture & Smoothness
    smoothness_ratio: float = 0.0
    laplacian_variance: float = 0.0

    # Error Level Analysis (ELA)
    ela_mean: float = 0.0
    ela_std: float = 0.0
    ela_max: float = 0.0

    # Optical Physics & Color Channels
    color_corr_rg: float = 0.0
    color_corr_rb: float = 0.0
    chromatic_dispersion: float = 0.0
    histogram_smoothness: float = 0.0

    # Combined Forensic Probability (0.0 to 1.0)
    forensic_score: float = 0.0
    anomalies: List[str] = field(default_factory=list)


class ForensicAnalyzer:
    """Vectorized mathematical and signal-level forensic engine."""

    @staticmethod
    def analyze(file_path: str) -> ForensicResult:
        if not os.path.isfile(file_path):
            return ForensicResult()

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            try:
                pil_img = Image.open(file_path).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                return ForensicResult()

        h, w, c = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        res = ForensicResult(width=w, height=h)

        # 1. Aspect ratio & AI canvas resolution check
        gcd = np.gcd(w, h)
        res.aspect_ratio = f"{w // gcd}:{h // gcd}" if gcd > 0 else f"{w}:{h}"
        if (w, h) in COMMON_AI_RESOLUTIONS:
            res.is_common_ai_resolution = True
            res.ai_resolution_desc = COMMON_AI_RESOLUTIONS[(w, h)]

        # 2. Vectorized 2D FFT Frequency Analysis (<50ms)
        ForensicAnalyzer._analyze_fft_fast(gray, res)

        # 3. Vectorized Noise Residual & Kurtosis
        ForensicAnalyzer._analyze_noise_fast(gray, res)

        # 4. Vectorized Local Texture Variance (<30ms)
        ForensicAnalyzer._analyze_texture_fast(gray, res)

        # 5. Error Level Analysis (ELA)
        ForensicAnalyzer._analyze_ela_fast(file_path, res)

        # 6. Optical Physics: Chromatic Dispersion & Histogram Smoothness
        ForensicAnalyzer._analyze_optics_and_histogram(img_bgr, gray, res)

        # 7. Aggregate Normalized Forensic Probability
        ForensicAnalyzer._calculate_forensic_score(res)

        return res

    @staticmethod
    def _analyze_fft_fast(gray: np.ndarray, res: ForensicResult) -> None:
        """Fast vectorized 2D FFT power spectrum with binned radial decay."""
        try:
            # Resize to 512x512 for lightning-fast, resolution-independent spectral profile
            target_dim = 512
            g_scaled = cv2.resize(gray, (target_dim, target_dim), interpolation=cv2.INTER_AREA)

            dft = np.fft.fft2(g_scaled.astype(np.float32))
            dft_shift = np.fft.fftshift(dft)
            psd = np.abs(dft_shift) ** 2

            cy, cx = target_dim // 2, target_dim // 2
            y, x = np.ogrid[:target_dim, :target_dim]
            r = np.hypot(x - cx, y - cy).astype(np.int32).ravel()

            rmax = target_dim // 2
            counts = np.bincount(r, minlength=rmax + 1)[:rmax]
            sums = np.bincount(r, weights=psd.ravel(), minlength=rmax + 1)[:rmax]

            valid = counts > 0
            radial = np.zeros(rmax)
            radial[valid] = sums[valid] / counts[valid]

            freqs = np.arange(rmax)
            p_valid = (freqs > 5) & (radial > 0)
            log_f = np.log10(freqs[p_valid])
            log_p = np.log10(radial[p_valid])

            # Fit power-law slope between 10% and 75% frequency
            s_idx = int(len(log_f) * 0.10)
            e_idx = int(len(log_f) * 0.75)

            if e_idx > s_idx + 5:
                fit_x = log_f[s_idx:e_idx]
                fit_y = log_p[s_idx:e_idx]
                slope, intercept = np.polyfit(fit_x, fit_y, 1)
                res.psd_slope = float(slope)

                y_pred = slope * fit_x + intercept
                ss_res = np.sum((fit_y - y_pred) ** 2)
                ss_tot = np.sum((fit_y - np.mean(fit_y)) ** 2)
                res.psd_r2_fit = float(1.0 - (ss_res / (ss_tot + 1e-9)))

            # High-frequency power ratio
            outer_mask = (r[:rmax] > (0.6 * rmax))
            total_e = np.sum(psd)
            if total_e > 0:
                res.high_freq_energy_ratio = float(np.sum(sums[outer_mask]) / total_e)

            # Check for periodic grid spikes (checkerboard upsampler artifacts)
            log_mag = np.log(np.abs(dft_shift) + 1.0)
            mid = log_mag[cy - rmax // 2 : cy + rmax // 2, cx - rmax // 2 : cx + rmax // 2]
            res.spectral_grid_score = float((np.max(mid) - np.mean(mid)) / (np.std(mid) + 1e-9))

        except Exception:
            pass

    @staticmethod
    def _analyze_noise_fast(gray: np.ndarray, res: ForensicResult) -> None:
        """Fast high-pass noise extraction and residual kurtosis."""
        try:
            blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
            residual = gray.astype(np.float32) - blur.astype(np.float32)
            flat = residual.ravel()

            res.noise_variance = float(np.var(flat))
            res.noise_kurtosis = float(scipy.stats.kurtosis(flat))
            res.noise_skewness = float(scipy.stats.skew(flat))
        except Exception:
            pass

    @staticmethod
    def _analyze_texture_fast(gray: np.ndarray, res: ForensicResult) -> None:
        """Vectorized box-filter local variance (1000x faster than sliding loops)."""
        try:
            res.laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Local variance in 16x16 window via integral filtering: E[X^2] - (E[X])^2
            ksize = 16
            g_f = gray.astype(np.float32)
            mean = cv2.boxFilter(g_f, -1, (ksize, ksize))
            sqr_mean = cv2.boxFilter(g_f * g_f, -1, (ksize, ksize))
            local_var = np.maximum(sqr_mean - (mean * mean), 0.0)

            # Ratio of patches with very low variance (< 4.0 implies completely airbrushed)
            res.smoothness_ratio = float(np.mean(local_var < 4.0))
        except Exception:
            pass

    @staticmethod
    def _analyze_ela_fast(file_path: str, res: ForensicResult) -> None:
        """Fast Error Level Analysis (recompression discrepancy)."""
        try:
            with Image.open(file_path) as orig_img:
                orig_rgb = orig_img.convert("RGB")
                buf = io.BytesIO()
                orig_rgb.save(buf, "JPEG", quality=90)
                buf.seek(0)
                recomp = Image.open(buf)

                arr_orig = np.array(orig_rgb, dtype=np.float32)
                arr_recomp = np.array(recomp, dtype=np.float32)

                diff = np.abs(arr_orig - arr_recomp)
                res.ela_mean = float(np.mean(diff))
                res.ela_std = float(np.std(diff))
                res.ela_max = float(np.max(diff))
        except Exception:
            pass

    @staticmethod
    def _analyze_optics_and_histogram(
        img_bgr: np.ndarray, gray: np.ndarray, res: ForensicResult
    ) -> None:
        """Analyzes chromatic dispersion and color histogram smoothness."""
        try:
            b, g, r = cv2.split(img_bgr)
            sob_b = cv2.Sobel(b, cv2.CV_32F, 1, 1).ravel()
            sob_g = cv2.Sobel(g, cv2.CV_32F, 1, 1).ravel()
            sob_r = cv2.Sobel(r, cv2.CV_32F, 1, 1).ravel()

            res.color_corr_rg = float(np.corrcoef(sob_r, sob_g)[0, 1])
            res.color_corr_rb = float(np.corrcoef(sob_r, sob_b)[0, 1])

            # Chromatic edge dispersion index
            mag_r = cv2.magnitude(cv2.Sobel(r, cv2.CV_32F, 1, 0), cv2.Sobel(r, cv2.CV_32F, 0, 1))
            mag_b = cv2.magnitude(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1))
            strong = mag_r > np.percentile(mag_r, 90)
            if np.any(strong):
                disp = np.mean(np.abs(mag_r[strong] - mag_b[strong])) / (np.mean(mag_r[strong]) + 1e-6)
                res.chromatic_dispersion = float(disp)

            # Histogram smoothness (diffusion outputs generate hyper-smooth 2nd derivatives)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
            hist_norm = hist / (np.sum(hist) + 1e-9)
            res.histogram_smoothness = float(np.std(np.diff(hist_norm, n=2)) * 1000.0)

        except Exception:
            pass

    @staticmethod
    def _calculate_forensic_score(res: ForensicResult) -> None:
        """Weighted multi-domain anomaly synthesis."""
        score = 0.0
        anomalies = []

        # 1. Noise Kurtosis (Normal camera: -1.0 to 6.0)
        if res.noise_kurtosis > 25.0:
            score += 0.35
            anomalies.append(f"Extreme noise kurtosis ({res.noise_kurtosis:.1f} > 25): synthetic heavy-tailed denoising residual")
        elif res.noise_kurtosis > 12.0:
            score += 0.20
            anomalies.append(f"Elevated noise kurtosis ({res.noise_kurtosis:.1f}): non-sensor noise pattern")
        elif res.noise_kurtosis < -1.0:
            score += 0.15
            anomalies.append("Unusually uniform/suppressed noise distribution")

        # 2. Radial PSD Power-Law Slope (Natural: -1.8 to -2.4)
        if res.psd_slope < -2.9:
            score += 0.25
            anomalies.append(f"Steep high-frequency spectral rolloff ({res.psd_slope:.2f}): unnatural loss of physical micro-texture")
        elif res.psd_slope > -1.5 and res.psd_slope != 0.0:
            score += 0.20
            anomalies.append(f"Abnormal flat spectral slope ({res.psd_slope:.2f}): high-frequency synthetic noise artifact")

        # 3. Spectral Grid / Peak Anomaly
        if res.spectral_grid_score > 7.5:
            score += 0.15
            anomalies.append(f"High spectral peak ratio ({res.spectral_grid_score:.1f}): periodic grid/deconvolution artifact detected")

        # 4. Airbrush / Smoothness Ratio
        if res.smoothness_ratio > 0.40:
            score += 0.20
            anomalies.append(f"High textureless patch ratio ({res.smoothness_ratio*100:.1f}%): heavy synthetic airbrushing / denoising")
        elif res.smoothness_ratio > 0.25:
            score += 0.10
            anomalies.append(f"Moderate textureless patch ratio ({res.smoothness_ratio*100:.1f}%)")

        # 5. Standard AI Canvas Dimensions
        if res.is_common_ai_resolution:
            score += 0.10
            anomalies.append(f"Canonical AI generator canvas size: {res.width}x{res.height} ({res.ai_resolution_desc})")

        # 6. ELA Dynamic Uniformity
        if res.ela_mean > 0 and res.ela_std > 0:
            ratio = res.ela_std / (res.ela_mean + 1e-9)
            if ratio < 0.6:
                score += 0.10
                anomalies.append("Uniform compression error profile across varied textures")

        # 7. Chromatic Dispersion Anomaly
        if res.chromatic_dispersion > 0.25:
            score += 0.10
            anomalies.append("Abnormal inter-channel edge dispersion inconsistent with optical glass")

        res.forensic_score = min(max(score, 0.0), 1.0)
        res.anomalies = anomalies
