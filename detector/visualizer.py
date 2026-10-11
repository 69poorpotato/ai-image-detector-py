"""
Forensic diagnostic visualizer.
Generates multi-panel comparison heatmaps (Original, 2D FFT Spectrum,
Error Level Analysis (ELA), and Noise Residual) to export as images.
"""

from __future__ import annotations
import io
import os
import cv2
import numpy as np
from PIL import Image


class ForensicVisualizer:
    """Generates visual forensic comparison cards and heatmaps."""

    @staticmethod
    def generate_diagnostic_map(
        file_path: str,
        output_path: str,
        panel_size: int = 512
    ) -> bool:
        """
        Creates a 4-panel diagnostic image:
        [1. Original Image]   [2. 2D FFT Power Spectrum]
        [3. ELA Heatmap]      [4. Noise Residual Map]
        """
        if not os.path.isfile(file_path):
            return False

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            return False

        # Panel 1: Original Image (resized square for uniform tiling)
        p1 = cv2.resize(img_bgr, (panel_size, panel_size))
        ForensicVisualizer._draw_label(p1, "1. ORIGINAL IMAGE")

        # Panel 2: 2D FFT Spectrum
        gray = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
        dft = np.fft.fft2(gray.astype(np.float64))
        dft_shift = np.fft.fftshift(dft)
        mag = 20 * np.log(np.abs(dft_shift) + 1.0)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        p2 = cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA)
        ForensicVisualizer._draw_label(p2, "2. 2D FFT POWER SPECTRUM")

        # Panel 3: Error Level Analysis (ELA)
        p3 = ForensicVisualizer._generate_ela_panel(file_path, panel_size)
        ForensicVisualizer._draw_label(p3, "3. ERROR LEVEL ANALYSIS (ELA)")

        # Panel 4: Noise Residual
        blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
        residual = np.abs(gray.astype(np.float64) - blur.astype(np.float64))
        res_norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        p4 = cv2.applyColorMap(res_norm, cv2.COLORMAP_VIRIDIS)
        ForensicVisualizer._draw_label(p4, "4. NOISE RESIDUAL MAP")

        # Combine into 2x2 grid
        top_row = np.hstack([p1, p2])
        bottom_row = np.hstack([p3, p4])
        grid = np.vstack([top_row, bottom_row])

        # Ensure directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(output_path, grid)
        return True

    @staticmethod
    def _generate_ela_panel(file_path: str, size: int) -> np.ndarray:
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
                # Boost brightness for visual clarity (15x scale)
                diff_boosted = np.clip(diff * 15.0, 0, 255).astype(np.uint8)
                diff_bgr = cv2.cvtColor(diff_boosted, cv2.COLOR_RGB2BGR)
                return cv2.resize(diff_bgr, (size, size))
        except Exception:
            blank = np.zeros((size, size, 3), dtype=np.uint8)
            return blank

    @staticmethod
    def _draw_label(img: np.ndarray, text: str) -> None:
        cv2.rectangle(img, (0, 0), (len(text) * 16 + 20, 32), (20, 20, 20), -1)
        cv2.putText(
            img,
            text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
