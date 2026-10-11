"""
AI Image Detector & Forensic Analyzer
A multi-layered tool combining C2PA Content Credentials, deep metadata parsing,
and signal/spectral forensic analysis to detect AI-generated images.
"""

from .analyzer import AIImageDetector, AnalysisReport, Verdict
from .c2pa_checker import C2PAChecker, C2PAResult
from .metadata_checker import MetadataChecker, MetadataResult
from .forensics import ForensicAnalyzer, ForensicResult
from .visualizer import ForensicVisualizer
from .html_reporter import HTMLReporter
from .ui import render_report, render_banner

__version__ = "1.1.0"
__all__ = [
    "AIImageDetector",
    "AnalysisReport",
    "Verdict",
    "C2PAChecker",
    "C2PAResult",
    "MetadataChecker",
    "MetadataResult",
    "ForensicAnalyzer",
    "ForensicResult",
    "ForensicVisualizer",
    "HTMLReporter",
    "render_report",
    "render_banner",
]
