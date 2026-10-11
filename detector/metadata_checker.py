"""
Metadata analysis module for detecting AI generation fingerprints,
prompts, workflow graphs, and camera EXIF markers.
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ExifTags


KNOWN_AI_SOFTWARE_SIGNATURES = [
    "automatic1111", "comfyui", "novelai", "invokeai", "fooocus",
    "midjourney", "firefly", "dall-e", "bing image creator", "copilot",
    "stable diffusion", "flux", "swarmui", "easyedit", "ideogram",
    "leonardo", "playground", "seaart", "krea", "sora", "runway",
    "pika", "luma", "imagen", "tensorart", "sd3", "sdxl"
]

CAMERA_MANUFACTURERS = [
    "apple", "canon", "nikon", "sony", "leica", "fujifilm",
    "panasonic", "olympus", "pentax", "samsung", "google", "hasselblad"
]


@dataclass
class MetadataResult:
    has_ai_metadata: bool = False
    detected_generator: Optional[str] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    model: Optional[str] = None
    sampler: Optional[str] = None
    steps: Optional[int] = None
    seed: Optional[int] = None
    cfg_scale: Optional[float] = None
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Camera provenance indicators
    has_camera_exif: bool = False
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    exposure_time: Optional[str] = None
    f_number: Optional[str] = None
    iso_speed: Optional[str] = None
    software_tag: Optional[str] = None
    
    raw_keys: list[str] = field(default_factory=list)
    raw_info: Dict[str, Any] = field(default_factory=dict)


class MetadataChecker:
    """Parses PNG chunks, EXIF, and XMP metadata to uncover AI prompts & camera tags."""

    @staticmethod
    def inspect(file_path: str) -> MetadataResult:
        if not os.path.isfile(file_path):
            return MetadataResult()

        result = MetadataResult()

        try:
            with Image.open(file_path) as img:
                info = getattr(img, "info", {}) or {}
                result.raw_info = {str(k): str(v)[:500] for k, v in info.items()}
                result.raw_keys = list(result.raw_info.keys())

                # 1. Parse PNG text chunks
                MetadataChecker._parse_png_chunks(img, info, result)

                # 2. Parse EXIF data
                MetadataChecker._parse_exif(img, result)

                # 3. Parse XMP if present
                MetadataChecker._parse_xmp(info, result)

        except Exception as e:
            result.generation_params["_error"] = str(e)

        return result

    @staticmethod
    def _parse_png_chunks(img: Image.Image, info: Dict[str, Any], res: MetadataResult) -> None:
        """Parses standard AI generation text chunks."""
        # Fooocus
        if "Fooocus" in info:
            res.has_ai_metadata = True
            res.detected_generator = "Fooocus"
            try:
                f_data = json.loads(info["Fooocus"])
                res.prompt = f_data.get("prompt")
                res.negative_prompt = f_data.get("negative_prompt")
                res.model = f_data.get("base_model")
            except Exception:
                pass

        # Automatic1111 / Forge / WebUI 'parameters'
        if "parameters" in info:
            res.has_ai_metadata = True
            param_text = str(info["parameters"])
            if "flux" in param_text.lower():
                res.detected_generator = "Flux.1 (via WebUI / Forge)"
            elif "sd3" in param_text.lower():
                res.detected_generator = "Stable Diffusion 3"
            elif "sdxl" in param_text.lower():
                res.detected_generator = "SDXL (Automatic1111 / WebUI)"
            else:
                res.detected_generator = res.detected_generator or "Stable Diffusion (Automatic1111 / WebUI)"
            MetadataChecker._parse_a1111_parameters(param_text, res)

        # ComfyUI 'prompt' or 'workflow'
        if "prompt" in info:
            try:
                prompt_json = json.loads(info["prompt"])
                res.has_ai_metadata = True
                res.detected_generator = res.detected_generator or "ComfyUI"
                MetadataChecker._parse_comfyui_prompt(prompt_json, res)
            except Exception:
                pass

        if "workflow" in info and not res.detected_generator:
            try:
                json.loads(info["workflow"])
                res.has_ai_metadata = True
                res.detected_generator = "ComfyUI"
            except Exception:
                pass

        # Ideogram / generic prompt chunk
        for k in ["prompt", "Prompt", "ai_prompt", "generation_prompt"]:
            if k in info and not res.prompt and isinstance(info[k], str) and len(info[k]) > 5:
                res.has_ai_metadata = True
                res.prompt = info[k].strip()
                res.detected_generator = res.detected_generator or "AI Generated Media"

        # NovelAI
        if "Comment" in info or "Description" in info:
            comment_str = str(info.get("Comment") or info.get("Description", ""))
            if comment_str.strip().startswith("{") and "prompt" in comment_str.lower():
                try:
                    c_json = json.loads(comment_str)
                    res.has_ai_metadata = True
                    res.detected_generator = res.detected_generator or "NovelAI"
                    res.prompt = c_json.get("prompt")
                    res.negative_prompt = c_json.get("uc") or c_json.get("negative_prompt")
                    res.steps = c_json.get("steps")
                    res.seed = c_json.get("seed")
                    res.sampler = c_json.get("sampler")
                except Exception:
                    pass

        # Check Software chunk
        software = str(info.get("Software", "")).strip()
        if software:
            res.software_tag = software
            sw_lower = software.lower()
            for sig in KNOWN_AI_SOFTWARE_SIGNATURES:
                if sig in sw_lower:
                    res.has_ai_metadata = True
                    res.detected_generator = software

    @staticmethod
    def _parse_a1111_parameters(text: str, res: MetadataResult) -> None:
        """Parses Automatic1111 formatted generation strings."""
        parts = text.split("Negative prompt:")
        res.prompt = parts[0].strip()

        remaining = parts[1] if len(parts) > 1 else ""
        neg_parts = remaining.split("Steps:")
        if len(neg_parts) > 1:
            res.negative_prompt = neg_parts[0].strip()
            param_line = "Steps:" + neg_parts[1]
        else:
            param_line = remaining

        # Parse key-value pairs (e.g. Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 12345, Model: v1-5)
        pattern = r'([A-Za-z0-9_ ]+):\s*([^,]+)(?:,|$)'
        for match in re.finditer(pattern, param_line):
            key = match.group(1).strip()
            val = match.group(2).strip()
            res.generation_params[key] = val

            key_lower = key.lower()
            if "step" in key_lower:
                try:
                    res.steps = int(val)
                except ValueError:
                    pass
            elif "sampler" in key_lower:
                res.sampler = val
            elif "seed" in key_lower and "subseed" not in key_lower:
                try:
                    res.seed = int(val)
                except ValueError:
                    pass
            elif "cfg" in key_lower:
                try:
                    res.cfg_scale = float(val)
                except ValueError:
                    pass
            elif "model" in key_lower and "hash" not in key_lower:
                res.model = val

    @staticmethod
    def _parse_comfyui_prompt(node_graph: Dict[str, Any], res: MetadataResult) -> None:
        """Extracts prompts and sampler info from a ComfyUI node graph."""
        prompts = []
        for node_id, node in node_graph.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})

            # Text prompt nodes
            if "CLIPTextEncode" in class_type or "PrimitiveString" in class_type:
                text_val = inputs.get("text")
                if text_val and isinstance(text_val, str) and len(text_val.strip()) > 2:
                    prompts.append(text_val.strip())

            # Sampler nodes
            if "KSampler" in class_type:
                if "steps" in inputs:
                    res.steps = inputs.get("steps")
                if "seed" in inputs:
                    res.seed = inputs.get("seed")
                if "cfg" in inputs:
                    res.cfg_scale = inputs.get("cfg")
                if "sampler_name" in inputs:
                    res.sampler = inputs.get("sampler_name")

            # Checkpoint loaders
            if "CheckpointLoader" in class_type or "UNETLoader" in class_type:
                ckpt = inputs.get("ckpt_name") or inputs.get("unet_name")
                if ckpt:
                    res.model = str(ckpt)

        if prompts and not res.prompt:
            res.prompt = prompts[0]
            if len(prompts) > 1:
                res.negative_prompt = prompts[1]

    @staticmethod
    def _parse_exif(img: Image.Image, res: MetadataResult) -> None:
        """Extracts camera hardware and generation flags from EXIF."""
        try:
            exif = img.getexif()
            if not exif:
                return

            exif_dict = {}
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif_dict[tag_name] = value

            # Camera provenance
            make = str(exif_dict.get("Make", "")).strip()
            model = str(exif_dict.get("Model", "")).strip()
            software = str(exif_dict.get("Software", "")).strip()
            user_comment = str(exif_dict.get("UserComment", "")).strip()

            if software:
                res.software_tag = software
                for sig in KNOWN_AI_SOFTWARE_SIGNATURES:
                    if sig in software.lower():
                        res.has_ai_metadata = True
                        res.detected_generator = res.detected_generator or software

            if user_comment:
                for sig in KNOWN_AI_SOFTWARE_SIGNATURES:
                    if sig in user_comment.lower():
                        res.has_ai_metadata = True
                        res.detected_generator = res.detected_generator or "AI Generated (UserComment)"
                if not res.prompt and len(user_comment) > 10:
                    res.prompt = user_comment[:400]

            if make or model:
                make_lower = make.lower()
                if any(m in make_lower for m in CAMERA_MANUFACTURERS):
                    res.has_camera_exif = True
                    res.camera_make = make
                    res.camera_model = model

            # Check detailed lens and exposure settings
            lens = str(exif_dict.get("LensModel", "")).strip()
            if lens:
                res.lens_model = lens

            focal = exif_dict.get("FocalLength")
            fnumber = exif_dict.get("FNumber")
            exposure = exif_dict.get("ExposureTime")
            iso = exif_dict.get("ISOSpeedRatings")

            if fnumber:
                res.f_number = f"f/{fnumber}"
            if exposure:
                res.exposure_time = f"{exposure}s"
            if iso:
                res.iso_speed = f"ISO {iso}"

            if (res.camera_make or res.camera_model) and (res.f_number or res.exposure_time or res.iso_speed):
                res.has_camera_exif = True

        except Exception:
            pass

    @staticmethod
    def _parse_xmp(info: Dict[str, Any], res: MetadataResult) -> None:
        """Parses XMP tags for CreatorTool and AI digital source tags."""
        xmp_raw = info.get("XML:com.adobe.xmp") or info.get("xmp")
        if not xmp_raw:
            return

        xmp_str = str(xmp_raw)
        for sig in KNOWN_AI_SOFTWARE_SIGNATURES:
            if sig in xmp_str.lower():
                res.has_ai_metadata = True
                res.detected_generator = res.detected_generator or f"XMP Tag: {sig.title()}"

        if "trainedalgorithmicmedia" in xmp_str.lower():
            res.has_ai_metadata = True
            res.detected_generator = res.detected_generator or "IPTC Trained Algorithmic Media"
