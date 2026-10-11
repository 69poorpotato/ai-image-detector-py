"""
Unit and integration test suite for AI Image Detector.
"""

import io
import os
import unittest
import numpy as np
from PIL import Image, PngImagePlugin

from detector import (
    AIImageDetector,
    C2PAChecker,
    ForensicAnalyzer,
    ForensicVisualizer,
    MetadataChecker,
    Verdict,
)


class TestAIImageDetector(unittest.TestCase):

    def setUp(self):
        self.detector = AIImageDetector()
        self.test_img_path = "image.png"

    def test_c2pa_image_png(self):
        """Verifies that image.png is correctly recognized as ChatGPT AI via C2PA."""
        if not os.path.isfile(self.test_img_path):
            self.skipTest("image.png not found")

        res = C2PAChecker.inspect(self.test_img_path)
        self.assertTrue(res.has_c2pa)
        self.assertTrue(res.is_confirmed_ai)
        self.assertIn("ChatGPT", res.detected_tool)
        self.assertEqual(res.issuer, "OpenAI OpCo, LLC")
        self.assertIn("c2pa.watermarked.unbound", res.watermarks)

    def test_forensics_image_png(self):
        """Verifies signal and frequency forensic metrics on image.png."""
        if not os.path.isfile(self.test_img_path):
            self.skipTest("image.png not found")

        f = ForensicAnalyzer.analyze(self.test_img_path)
        self.assertEqual(f.width, 1353)
        self.assertEqual(f.height, 1163)
        # Verify kurtosis is high for synthetic image
        self.assertGreater(f.noise_kurtosis, 15.0)
        # Verify forensic anomaly score is elevated
        self.assertGreater(f.forensic_score, 0.6)

    def test_metadata_extraction_synthetic(self):
        """Tests parsing of embedded Automatic1111 generation parameters."""
        dummy_path = "test_a1111_sample.png"
        img = Image.new("RGB", (512, 512), color="blue")
        meta = PngImagePlugin.PngInfo()
        meta.add_text(
            "parameters",
            "A cyberpunk hacker girl in neon rain\nNegative prompt: blurry, ugly\nSteps: 25, Sampler: DPM++ 2M Karras, CFG scale: 7.5, Seed: 987654, Size: 512x512, Model: sd_xl_base_1.0"
        )
        img.save(dummy_path, "PNG", pnginfo=meta)

        try:
            res = MetadataChecker.inspect(dummy_path)
            self.assertTrue(res.has_ai_metadata)
            self.assertEqual(res.prompt, "A cyberpunk hacker girl in neon rain")
            self.assertEqual(res.negative_prompt, "blurry, ugly")
            self.assertEqual(res.steps, 25)
            self.assertEqual(res.sampler, "DPM++ 2M Karras")
            self.assertEqual(res.seed, 987654)
            self.assertEqual(res.cfg_scale, 7.5)
            self.assertEqual(res.model, "sd_xl_base_1.0")
        finally:
            if os.path.isfile(dummy_path):
                os.remove(dummy_path)

    def test_full_analysis_report(self):
        """Tests full analysis pipeline on image.png."""
        if not os.path.isfile(self.test_img_path):
            self.skipTest("image.png not found")

        report = self.detector.analyze(self.test_img_path)
        self.assertEqual(report.verdict, Verdict.CONFIRMED_AI_PROVENANCE)
        self.assertEqual(report.confidence, 100.0)
        self.assertIsNotNone(report.detected_generator)

        # Test dictionary export
        data = report.to_dict()
        self.assertEqual(data["confidence"], 100.0)
        self.assertIn("c2pa", data)
        self.assertIn("forensics", data)

    def test_visualizer_generation(self):
        """Tests generation of 4-panel diagnostic heatmap."""
        if not os.path.isfile(self.test_img_path):
            self.skipTest("image.png not found")

        out_map = "test_diagnostic_out.png"
        try:
            ok = ForensicVisualizer.generate_diagnostic_map(self.test_img_path, out_map, panel_size=256)
            self.assertTrue(ok)
            self.assertTrue(os.path.isfile(out_map))
            self.assertGreater(os.path.getsize(out_map), 10000)
        finally:
            if os.path.isfile(out_map):
                os.remove(out_map)

    def test_html_report_generation(self):
        """Tests generation of standalone interactive HTML report."""
        if not os.path.isfile(self.test_img_path):
            self.skipTest("image.png not found")

        out_html = "test_audit_out.html"
        try:
            report = self.detector.analyze(self.test_img_path)
            from detector import HTMLReporter
            ok = HTMLReporter.generate(report, out_html)
            self.assertTrue(ok)
            self.assertTrue(os.path.isfile(out_html))
            self.assertGreater(os.path.getsize(out_html), 1000)
        finally:
            if os.path.isfile(out_html):
                os.remove(out_html)


if __name__ == "__main__":
    unittest.main()
