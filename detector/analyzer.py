"""
Master analysis engine orchestrating C2PA, metadata, and signal forensics
to reach an integrated verdict and confidence score.
"""

from __future__ import annotations
import hashlib
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from PIL import Image

from .c2pa_checker import C2PAChecker, C2PAResult
from .metadata_checker import MetadataChecker, MetadataResult
from .forensics import ForensicAnalyzer, ForensicResult


class Verdict(str, Enum):
    CONFIRMED_AI_PROVENANCE = "CONFIRMED AI (C2PA / Content Credentials)"
    CONFIRMED_AI_METADATA = "CONFIRMED AI (Generation Metadata / Prompts)"
    HIGH_PROBABILITY_AI = "HIGH PROBABILITY AI (Forensic Anomaly)"
    SUSPICIOUS_AI = "SUSPICIOUS / LIKELY AI (Signal Irregularity)"
    INCONCLUSIVE = "INCONCLUSIVE (Metadata Stripped / Ambiguous)"
    LIKELY_AUTHENTIC_CAMERA = "LIKELY AUTHENTIC (Camera Hardware Markers)"
    AUTHENTIC_VERIFIED_CAMERA = "VERIFIED AUTHENTIC (C2PA Camera Provenance)"


@dataclass
class FileInfo:
    file_path: str
    file_name: str
    file_size_bytes: int
    file_size_human: str
    format: str
    dimensions: str
    sha256: str


@dataclass
class AnalysisReport:
    file_info: FileInfo
    verdict: Verdict
    confidence: float  # 0.0 to 100.0
    primary_indicator: str
    detected_generator: Optional[str] = None
    key_findings: List[str] = field(default_factory=list)
    
    c2pa_result: Optional[C2PAResult] = None
    metadata_result: Optional[MetadataResult] = None
    forensic_result: Optional[ForensicResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts report to serializable dictionary."""
        return {
            "file_info": asdict(self.file_info),
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 1),
            "primary_indicator": self.primary_indicator,
            "detected_generator": self.detected_generator,
            "key_findings": self.key_findings,
            "c2pa": {
                "has_c2pa": self.c2pa_result.has_c2pa if self.c2pa_result else False,
                "is_confirmed_ai": self.c2pa_result.is_confirmed_ai if self.c2pa_result else False,
                "claim_generator": self.c2pa_result.claim_generator if self.c2pa_result else None,
                "issuer": self.c2pa_result.issuer if self.c2pa_result else None,
                "actions": [asdict(a) for a in self.c2pa_result.actions] if self.c2pa_result else [],
                "watermarks": self.c2pa_result.watermarks if self.c2pa_result else [],
            } if self.c2pa_result else {},
            "metadata": {
                "has_ai_metadata": self.metadata_result.has_ai_metadata if self.metadata_result else False,
                "detected_generator": self.metadata_result.detected_generator if self.metadata_result else None,
                "prompt": self.metadata_result.prompt if self.metadata_result else None,
                "negative_prompt": self.metadata_result.negative_prompt if self.metadata_result else None,
                "model": self.metadata_result.model if self.metadata_result else None,
                "seed": self.metadata_result.seed if self.metadata_result else None,
                "has_camera_exif": self.metadata_result.has_camera_exif if self.metadata_result else False,
                "camera_make": self.metadata_result.camera_make if self.metadata_result else None,
                "camera_model": self.metadata_result.camera_model if self.metadata_result else None,
            } if self.metadata_result else {},
            "forensics": {
                "forensic_score": round(self.forensic_result.forensic_score, 3) if self.forensic_result else 0.0,
                "noise_kurtosis": round(self.forensic_result.noise_kurtosis, 2) if self.forensic_result else 0.0,
                "psd_slope": round(self.forensic_result.psd_slope, 3) if self.forensic_result else 0.0,
                "spectral_grid_score": round(self.forensic_result.spectral_grid_score, 2) if self.forensic_result else 0.0,
                "smoothness_ratio": round(self.forensic_result.smoothness_ratio, 3) if self.forensic_result else 0.0,
                "anomalies": self.forensic_result.anomalies if self.forensic_result else [],
            } if self.forensic_result else {},
        }


class AIImageDetector:
    """Master multi-layered AI image detector."""

    def __init__(self, confidence_threshold: float = 50.0):
        self.threshold = confidence_threshold

    def analyze(self, file_path: str) -> AnalysisReport:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 1. Basic File Info & SHA256
        file_info = self._get_file_info(file_path)

        # 2. Layer 1: C2PA Provenance
        c2pa_res = C2PAChecker.inspect(file_path)

        # 3. Layer 2: Metadata & Prompts
        meta_res = MetadataChecker.inspect(file_path)

        # 4. Layer 3: Forensic & Signal Processing
        forensic_res = ForensicAnalyzer.analyze(file_path)

        # 5. Synthesize Verdict & Findings
        verdict, confidence, indicator, tool, findings = self._evaluate_signals(
            c2pa_res, meta_res, forensic_res
        )

        return AnalysisReport(
            file_info=file_info,
            verdict=verdict,
            confidence=confidence,
            primary_indicator=indicator,
            detected_generator=tool,
            key_findings=findings,
            c2pa_result=c2pa_res,
            metadata_result=meta_res,
            forensic_result=forensic_res,
        )

    def _get_file_info(self, file_path: str) -> FileInfo:
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.2f} MB"

        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        sha256_hash = hasher.hexdigest()

        fmt = "UNKNOWN"
        dims = "UNKNOWN"
        try:
            with Image.open(file_path) as im:
                fmt = im.format or "UNKNOWN"
                dims = f"{im.width} x {im.height}"
        except Exception:
            pass

        return FileInfo(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            file_size_bytes=size_bytes,
            file_size_human=size_str,
            format=fmt,
            dimensions=dims,
            sha256=sha256_hash,
        )

    def _evaluate_signals(
        self,
        c2pa_res: C2PAResult,
        meta_res: MetadataResult,
        forensic_res: ForensicResult,
    ) -> tuple[Verdict, float, str, Optional[str], List[str]]:
        findings: List[str] = []
        detected_tool: Optional[str] = None

        # --- Priority 1: C2PA Provenance ---
        if c2pa_res.has_c2pa:
            findings.append("C2PA Content Credentials manifest found and parsed.")
            if c2pa_res.claim_generator:
                findings.append(f"Claim Generator: {c2pa_res.claim_generator}")
            if c2pa_res.issuer:
                findings.append(f"Signed by certificate authority: {c2pa_res.issuer}")

            if c2pa_res.is_confirmed_camera:
                findings.append("C2PA assertion verifies authentic hardware camera capture.")
                return (
                    Verdict.AUTHENTIC_VERIFIED_CAMERA,
                    99.9,
                    "Cryptographically verified C2PA camera provenance",
                    None,
                    findings,
                )

            if c2pa_res.is_confirmed_ai:
                tool = c2pa_res.detected_tool or "AI Generative Model"
                findings.append(f"C2PA digitalSourceType confirms trained algorithmic media ({tool}).")
                if c2pa_res.watermarks:
                    findings.append(f"Embedded watermarks identified: {', '.join(c2pa_res.watermarks)}")
                return (
                    Verdict.CONFIRMED_AI_PROVENANCE,
                    100.0,
                    f"C2PA Content Credentials ({tool})",
                    tool,
                    findings,
                )

        # --- Priority 2: Direct Generation Metadata / Prompts ---
        if meta_res.has_ai_metadata:
            tool = meta_res.detected_generator or "AI Generator"
            findings.append(f"Embedded AI generation metadata found ({tool}).")
            if meta_res.prompt:
                findings.append(f"Extracted generation prompt: \"{meta_res.prompt[:120]}...\"")
            if meta_res.model:
                findings.append(f"Model checkpoint: {meta_res.model}")
            if meta_res.seed is not None:
                findings.append(f"Generation seed: {meta_res.seed}")

            return (
                Verdict.CONFIRMED_AI_METADATA,
                99.5,
                f"Generation metadata & parameters ({tool})",
                tool,
                findings,
            )

        # --- Priority 3: Camera EXIF Provenance vs Forensic Anomalies ---
        has_camera = meta_res.has_camera_exif
        if has_camera:
            findings.append(
                f"Camera EXIF tags present: {meta_res.camera_make or ''} {meta_res.camera_model or ''} (Lens: {meta_res.lens_model or 'Standard'})"
            )

        # Include forensic findings
        findings.extend(forensic_res.anomalies)

        f_score = forensic_res.forensic_score

        if has_camera and f_score < 0.40:
            return (
                Verdict.LIKELY_AUTHENTIC_CAMERA,
                max(80.0, 100.0 - (f_score * 50.0)),
                "Authentic camera hardware EXIF & natural sensor noise profile",
                None,
                findings,
            )

        if f_score >= 0.70:
            pct = round(f_score * 100.0, 1)
            return (
                Verdict.HIGH_PROBABILITY_AI,
                pct,
                f"Multiple forensic signals indicate synthetic pixel patterns ({pct}%)",
                None,
                findings,
            )

        if f_score >= 0.45:
            pct = round(f_score * 100.0, 1)
            return (
                Verdict.SUSPICIOUS_AI,
                pct,
                f"Signal irregularities detected in frequency and noise domains ({pct}%)",
                None,
                findings,
            )

        if has_camera:
            return (
                Verdict.LIKELY_AUTHENTIC_CAMERA,
                75.0,
                "Camera hardware markers present with minor processing artifacts",
                None,
                findings,
            )

        findings.append("No C2PA manifest or generator metadata detected.")
        findings.append("Signal forensics did not reach anomaly threshold.")
        return (
            Verdict.INCONCLUSIVE,
            30.0,
            "Metadata absent, pixel forensics inconclusive",
            None,
            findings,
        )
