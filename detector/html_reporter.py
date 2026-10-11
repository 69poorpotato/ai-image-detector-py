"""
Standalone interactive HTML report generator.
Produces a modern, responsive, self-contained HTML audit report
with embedded image previews, verdict gauges, and forensic diagnostics.
"""

from __future__ import annotations
import base64
import os
from typing import Optional
from PIL import Image
import io

from .analyzer import AnalysisReport, Verdict


class HTMLReporter:
    """Generates self-contained interactive HTML audit reports."""

    @staticmethod
    def generate(report: AnalysisReport, output_html_path: str) -> bool:
        try:
            f_info = report.file_info
            verdict = report.verdict
            conf = report.confidence

            # Generate base64 thumbnail of the image
            b64_img = ""
            if os.path.isfile(f_info.file_path):
                try:
                    with Image.open(f_info.file_path) as im:
                        im_thumb = im.copy()
                        im_thumb.thumbnail((600, 600))
                        buf = io.BytesIO()
                        im_thumb.convert("RGB").save(buf, format="JPEG", quality=85)
                        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception:
                    pass

            # Badge styling
            if "CONFIRMED" in verdict.value or "HIGH" in verdict.value:
                badge_class = "badge-danger"
                verdict_title = "AI GENERATED"
            elif "SUSPICIOUS" in verdict.value:
                badge_class = "badge-warning"
                verdict_title = "SUSPICIOUS / LIKELY AI"
            elif "AUTHENTIC" in verdict.value:
                badge_class = "badge-success"
                verdict_title = "AUTHENTIC PHOTO"
            else:
                badge_class = "badge-neutral"
                verdict_title = "INCONCLUSIVE"

            # Build findings HTML
            findings_html = "".join([f"<li>{item}</li>" for item in report.key_findings])

            # Build C2PA section if present
            c2pa_html = ""
            if report.c2pa_result and report.c2pa_result.has_c2pa:
                c = report.c2pa_result
                actions_html = "".join([
                    f"<tr><td>{a.action}</td><td>{a.software_name or 'N/A'} {a.software_version or ''}</td><td>{a.digital_source_label or 'N/A'}</td></tr>"
                    for a in c.actions
                ])
                c2pa_html = f"""
                <div class="card">
                    <h3>Content Credentials & C2PA Provenance</h3>
                    <table>
                        <tr><th>Property</th><th>Value</th></tr>
                        <tr><td>Media Engine</td><td>{c.claim_generator or 'N/A'}</td></tr>
                        <tr><td>Signing Authority</td><td>{c.issuer or 'N/A'} ({c.common_name or ''})</td></tr>
                        <tr><td>Timestamp</td><td>{c.signature_time or 'N/A'}</td></tr>
                        <tr><td>Watermark Assertions</td><td>{', '.join(c.watermarks) if c.watermarks else 'None'}</td></tr>
                    </table>
                    <h4 style="margin-top:15px;">Actions Audit Trail</h4>
                    <table>
                        <tr><th>Action</th><th>Software</th><th>Source Type</th></tr>
                        {actions_html}
                    </table>
                </div>
                """

            # Build Metadata / Prompt section if present
            meta_html = ""
            if report.metadata_result and (report.metadata_result.has_ai_metadata or report.metadata_result.has_camera_exif):
                m = report.metadata_result
                prompt_box = ""
                if m.prompt:
                    prompt_box = f"""
                    <div class="prompt-box">
                        <div class="prompt-label">Generation Prompt:</div>
                        <div class="prompt-text">"{m.prompt}"</div>
                        {f'<div class="neg-prompt">Negative Prompt: "{m.negative_prompt}"</div>' if m.negative_prompt else ''}
                    </div>
                    """
                meta_html = f"""
                <div class="card">
                    <h3>Embedded Generation Metadata</h3>
                    {prompt_box}
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        {f'<tr><td>Detected Generator</td><td>{m.detected_generator}</td></tr>' if m.detected_generator else ''}
                        {f'<tr><td>Model Checkpoint</td><td>{m.model}</td></tr>' if m.model else ''}
                        {f'<tr><td>Sampler</td><td>{m.sampler}</td></tr>' if m.sampler else ''}
                        {f'<tr><td>Steps</td><td>{m.steps}</td></tr>' if m.steps else ''}
                        {f'<tr><td>CFG Scale</td><td>{m.cfg_scale}</td></tr>' if m.cfg_scale else ''}
                        {f'<tr><td>Seed</td><td>{m.seed}</td></tr>' if m.seed is not None else ''}
                        {f'<tr><td>Camera Hardware</td><td>{m.camera_make} {m.camera_model}</td></tr>' if m.has_camera_exif else ''}
                    </table>
                </div>
                """

            # Build Forensics section
            forensics_html = ""
            if report.forensic_result:
                f = report.forensic_result
                f_score = int(f.forensic_score * 100)
                forensics_html = f"""
                <div class="card">
                    <h3>Signal & Pixel Forensics (Anomaly Score: {f_score}%)</h3>
                    <table>
                        <tr><th>Forensic Metric</th><th>Observed Status</th><th>Reference Baseline</th></tr>
                        <tr><td>Texture & Skin Airbrushing</td><td><span class="badge { 'badge-danger' if f.smoothness_ratio > 0.3 else 'badge-success' }">{ 'Synthetic Smoothing' if f.smoothness_ratio > 0.3 else 'Natural Texture' }</span></td><td>Micro-texture variance</td></tr>
                        <tr><td>Noise Residual Kurtosis</td><td><span class="badge { 'badge-danger' if f.noise_kurtosis > 20 else 'badge-success' }">{f.noise_kurtosis:.2f} ({ 'High-tail Anomaly' if f.noise_kurtosis > 20 else 'Natural' })</span></td><td>-1.0 to +6.0 (Gaussian)</td></tr>
                        <tr><td>FFT Spectral Slope (α)</td><td><span class="badge { 'badge-danger' if f.psd_slope < -2.9 else 'badge-success' }">{f.psd_slope:.3f}</span></td><td>-1.8 to -2.4 (Power-law decay)</td></tr>
                        <tr><td>Spectral Grid Peak Ratio</td><td><span class="badge { 'badge-danger' if f.spectral_grid_score > 7 else 'badge-success' }">{f.spectral_grid_score:.2f}</span></td><td>&lt; 6.0 (Continuous)</td></tr>
                        <tr><td>Canvas Resolution</td><td>{f.width}x{f.height}</td><td>{f.ai_resolution_desc or 'Custom / Natural aspect'}</td></tr>
                    </table>
                </div>
                """

            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Detection Audit - {f_info.file_name}</title>
    <style>
        :root {{
            --bg: #0b0f19;
            --card-bg: #131b2e;
            --border: #1e293b;
            --text: #f1f5f9;
            --text-dim: #94a3b8;
            --cyan: #06b6d4;
            --red: #ef4444;
            --green: #10b981;
            --yellow: #f59e0b;
            --magenta: #d946ef;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background: var(--bg); color: var(--text); padding: 30px 20px; line-height: 1.6; }}
        .container {{ max-width: 950px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 30px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }}
        header h1 {{ font-size: 26px; color: var(--cyan); margin-bottom: 6px; letter-spacing: -0.5px; }}
        header p {{ color: var(--text-dim); font-size: 14px; }}
        .grid-summary {{ display: grid; grid-template-columns: 280px 1fr; gap: 24px; margin-bottom: 24px; }}
        @media(max-width: 768px) {{ .grid-summary {{ grid-template-columns: 1fr; }} }}
        .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 22px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .img-card {{ text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
        .img-card img {{ max-width: 100%; max-height: 260px; border-radius: 8px; border: 1px solid var(--border); object-fit: contain; }}
        .badge {{ display: inline-block; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: var(--red); border: 1px solid var(--red); }}
        .badge-success {{ background: rgba(16, 185, 129, 0.2); color: var(--green); border: 1px solid var(--green); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: var(--yellow); border: 1px solid var(--yellow); }}
        .badge-neutral {{ background: rgba(148, 163, 184, 0.2); color: var(--text-dim); border: 1px solid var(--text-dim); }}
        .progress-bar-bg {{ background: #1e293b; border-radius: 999px; height: 12px; overflow: hidden; margin-top: 8px; }}
        .progress-bar-fill {{ background: linear-gradient(90deg, var(--cyan), var(--yellow), var(--red)); height: 100%; border-radius: 999px; width: {conf}%; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #1e293b; }}
        th {{ color: var(--cyan); font-weight: 600; }}
        td {{ color: var(--text); }}
        ul {{ padding-left: 20px; margin-top: 10px; color: var(--text); }}
        li {{ margin-bottom: 6px; }}
        .prompt-box {{ background: #0f172a; border-left: 4px solid var(--cyan); padding: 14px; border-radius: 4px; margin-bottom: 14px; }}
        .prompt-label {{ font-size: 12px; color: var(--cyan); font-weight: bold; text-transform: uppercase; margin-bottom: 4px; }}
        .prompt-text {{ font-style: italic; color: #fff; font-size: 14px; }}
        .neg-prompt {{ color: var(--red); font-size: 13px; margin-top: 8px; }}
        footer {{ text-align: center; color: var(--text-dim); font-size: 12px; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI IMAGE DETECTOR & FORENSIC SUITE</h1>
            <p>Cryptographic Content Credentials • Metadata Parsing • Signal & Spectral Forensics</p>
        </header>

        <div class="grid-summary">
            <div class="card img-card">
                {f'<img src="data:image/jpeg;base64,{b64_img}" alt="{f_info.file_name}">' if b64_img else '<p style="color:var(--text-dim);">No preview</p>'}
                <p style="margin-top:10px; font-size:13px; color:var(--text-dim);">{f_info.file_name} ({f_info.dimensions})</p>
            </div>
            <div class="card">
                <div style="margin-bottom:14px;">
                    <span class="badge {badge_class}">{verdict_title}</span>
                </div>
                <div style="margin-bottom:12px;">
                    <span style="font-size:22px; font-weight:bold;">{report.detected_generator or verdict.value}</span>
                </div>
                <div style="margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; font-size:14px;">
                        <span>Confidence:</span>
                        <strong style="color:var(--yellow);">{conf:.1f}%</strong>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill"></div>
                    </div>
                </div>
                <p style="font-size:14px; color:var(--text-dim);"><strong>Primary Indicator:</strong> {report.primary_indicator}</p>
                <p style="font-size:12px; color:var(--text-dim); margin-top:6px;"><strong>SHA-256:</strong> <code>{f_info.sha256}</code></p>
            </div>
        </div>

        <div class="card">
            <h3>Evidence & Findings Summary</h3>
            <ul>
                {findings_html}
            </ul>
        </div>

        {c2pa_html}
        {meta_html}
        {forensics_html}

        <footer>
            Generated by AI Image Detector & Forensic Suite • Open Source MIT License
        </footer>
    </div>
</body>
</html>
"""
            out_dir = os.path.dirname(output_html_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            with open(output_html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            return True

        except Exception:
            return False
