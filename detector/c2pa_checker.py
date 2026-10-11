"""
C2PA / Content Credentials verification module.
Parses C2PA JUMBF manifests, digital source types, software agents,
signatures, and watermark assertions.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import c2pa
    HAS_C2PA = True
except ImportError:
    HAS_C2PA = False


AI_DIGITAL_SOURCE_TYPES = {
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia": "Trained Algorithmic Media (Pure AI Generation)",
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia": "Composite with AI-generated media",
    "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia": "Algorithmic Media (Computer-generated)",
    "trainedAlgorithmicMedia": "Trained Algorithmic Media (Pure AI Generation)",
    "compositeWithTrainedAlgorithmicMedia": "Composite with AI Media",
}

CAMERA_SOURCE_TYPES = {
    "http://cv.iptc.org/newscodes/digitalsourcetype/captured": "Optical Camera Capture (Hardware Sensor)",
    "captured": "Optical Camera Capture",
}

KNOWN_AI_GENERATOR_KEYWORDS = [
    "chatgpt", "openai", "dall-e", "dalle", "midjourney", "firefly",
    "stable diffusion", "bing image creator", "copilot", "google imagen",
    "flux", "sdxl", "novelai", "runway", "pika", "sora"
]


@dataclass
class C2PAAction:
    action: str
    software_name: Optional[str] = None
    software_version: Optional[str] = None
    digital_source_type: Optional[str] = None
    digital_source_label: Optional[str] = None
    timestamp: Optional[str] = None
    is_ai: bool = False


@dataclass
class C2PAResult:
    has_c2pa: bool = False
    title: Optional[str] = None
    claim_generator: Optional[str] = None
    claim_version: Optional[str] = None
    issuer: Optional[str] = None
    common_name: Optional[str] = None
    signature_time: Optional[str] = None
    validation_state: Optional[str] = None
    actions: List[C2PAAction] = field(default_factory=list)
    watermarks: List[str] = field(default_factory=list)
    raw_assertions: List[Dict[str, Any]] = field(default_factory=list)
    is_confirmed_ai: bool = False
    is_confirmed_camera: bool = False
    detected_tool: Optional[str] = None
    error_message: Optional[str] = None
    raw_manifest: Optional[Dict[str, Any]] = None


class C2PAChecker:
    """Checks images for C2PA Content Credentials manifests."""

    @staticmethod
    def inspect(file_path: str) -> C2PAResult:
        """Inspects an image file for C2PA provenance manifests."""
        if not HAS_C2PA:
            return C2PAResult(
                has_c2pa=False,
                error_message="c2pa-python library is not installed."
            )

        if not os.path.isfile(file_path):
            return C2PAResult(
                has_c2pa=False,
                error_message=f"File not found: {file_path}"
            )

        reader = None
        try:
            # First attempt: auto-detected stream
            with open(file_path, "rb") as f:
                reader = c2pa.Reader.try_create(None, stream=f)
                if reader is None:
                    return C2PAResult(has_c2pa=False)

                manifest_json_str = reader.json()
                data = json.loads(manifest_json_str)

            return C2PAChecker._parse_manifest_data(data)

        except Exception as e:
            err_str = str(e)
            if "ManifestNotFound" in err_str:
                return C2PAResult(has_c2pa=False)
            return C2PAResult(
                has_c2pa=False,
                error_message=f"C2PA inspection error: {err_str}"
            )
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

    @staticmethod
    def _parse_manifest_data(data: Dict[str, Any]) -> C2PAResult:
        result = C2PAResult(has_c2pa=True, raw_manifest=data)

        active_id = data.get("active_manifest")
        manifests = data.get("manifests", {})
        active_manifest = manifests.get(active_id, {})

        result.title = active_manifest.get("title")
        result.validation_state = data.get("validation_state", "Unknown")

        # Claim Generator info
        claim_gen_list = active_manifest.get("claim_generator_info", [])
        if claim_gen_list and isinstance(claim_gen_list, list):
            first_gen = claim_gen_list[0]
            result.claim_generator = first_gen.get("name")
            result.claim_version = first_gen.get("specVersion")
        else:
            result.claim_generator = active_manifest.get("claim_generator")

        # Signature info
        sig_info = active_manifest.get("signature_info", {})
        if sig_info:
            result.issuer = sig_info.get("issuer")
            result.common_name = sig_info.get("common_name")
            result.signature_time = sig_info.get("time")

        # Assertions
        assertions = active_manifest.get("assertions", [])
        result.raw_assertions = assertions

        for a in assertions:
            lbl = a.get("label", "")
            body = a.get("data", {})

            # Check actions (v1 or v2)
            if "actions" in lbl:
                actions_list = body.get("actions", [])
                for act in actions_list:
                    action_type = act.get("action", "")
                    sw_agent = act.get("softwareAgent", {})
                    if isinstance(sw_agent, dict):
                        sw_name = sw_agent.get("name")
                        sw_ver = sw_agent.get("version")
                    elif isinstance(sw_agent, str):
                        sw_name = sw_agent
                        sw_ver = None
                    else:
                        sw_name = None
                        sw_ver = None

                    src_type = act.get("digitalSourceType")
                    src_label = AI_DIGITAL_SOURCE_TYPES.get(src_type) or CAMERA_SOURCE_TYPES.get(src_type)

                    is_ai = False
                    if src_type in AI_DIGITAL_SOURCE_TYPES:
                        is_ai = True
                    if sw_name and any(k in sw_name.lower() for k in KNOWN_AI_GENERATOR_KEYWORDS):
                        is_ai = True

                    action_obj = C2PAAction(
                        action=action_type,
                        software_name=sw_name,
                        software_version=sw_ver,
                        digital_source_type=src_type,
                        digital_source_label=src_label,
                        timestamp=act.get("when"),
                        is_ai=is_ai
                    )
                    result.actions.append(action_obj)

                    if "watermarked" in action_type:
                        result.watermarks.append(action_type)

                    if is_ai:
                        result.is_confirmed_ai = True
                        if sw_name:
                            result.detected_tool = f"{sw_name} {sw_ver or ''}".strip()

                    if src_type in CAMERA_SOURCE_TYPES:
                        result.is_confirmed_camera = True

            # Direct AI / training assertions
            if "ai" in lbl.lower() or "generative" in lbl.lower():
                result.is_confirmed_ai = True

        # Secondary check on issuer / claim generator for AI services
        if not result.detected_tool:
            for item in [result.claim_generator, result.issuer, result.common_name]:
                if item:
                    item_lower = item.lower()
                    for k in KNOWN_AI_GENERATOR_KEYWORDS:
                        if k in item_lower:
                            result.detected_tool = item
                            # If OpenAI or similar service signed it with a create action
                            if any(act.action == "c2pa.created" for act in result.actions):
                                result.is_confirmed_ai = True
                            break

        return result
