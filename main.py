#!/usr/bin/env python3
"""
AI Image Detector & Forensic Suite CLI
Usage:
    python main.py <image_path>
    python main.py image.png --html report.html
    python main.py image.png --save-forensics diagnostic.png
    python main.py -i  (interactive drag-and-drop mode)
"""

import argparse
import glob
import json
import os
import sys
from typing import List, Optional

# Fix Windows console escape codes (fixes <-[1;36m in cmd.exe / PowerShell)
if sys.platform == "win32":
    try:
        import colorama
        colorama.just_fix_windows_console()
    except Exception:
        pass
    try:
        os.system("")
    except Exception:
        pass
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from detector import (
    AIImageDetector,
    ForensicVisualizer,
    HTMLReporter,
    Verdict,
    render_banner,
    render_report,
)
from rich.console import Console
from rich.table import Table
from rich import box

console = Console(width=84, highlight=False)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".avif"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Image Detector - Multi-layer verification for AI-generated media",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py image.png
  python main.py image.png --html report.html
  python main.py image.png --save-forensics diagnostic.png
  python main.py --dir ./photos -o scan.json
  python main.py -i  (interactive drag & drop mode)
        """
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Path to one or more image files to analyze. Defaults to 'image.png' if present.",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive drag-and-drop prompt",
    )
    parser.add_argument(
        "-d", "--dir",
        dest="directory",
        help="Scan all supported images in a directory",
    )
    parser.add_argument(
        "-s", "--save-forensics",
        nargs="?",
        const="forensics_diagnostic.png",
        metavar="OUTPUT_IMAGE",
        help="Generate and save 4-panel diagnostic visualizer map (FFT, ELA, Noise Residual)",
    )
    parser.add_argument(
        "--html",
        nargs="?",
        const="audit_report.html",
        metavar="OUTPUT_HTML",
        help="Generate self-contained interactive HTML audit report",
    )
    parser.add_argument(
        "-o", "--output",
        help="Export analysis report as a JSON file",
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        dest="json_output",
        help="Output raw JSON to stdout (suitable for automation/pipelines)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=50.0,
        help="Custom confidence threshold percentage (default: 50.0)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Display detailed mathematical parameters and raw logs",
    )

    return parser.parse_args()


def collect_image_paths(args: argparse.Namespace) -> List[str]:
    paths: List[str] = []

    if args.directory:
        if not os.path.isdir(args.directory):
            console.print(f"[bold red]Error:[/] Directory '{args.directory}' does not exist.")
            sys.exit(1)
        for root, _, files in os.walk(args.directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    paths.append(os.path.join(root, file))

    if args.images:
        for p in args.images:
            clean_p = p.strip('"\'')
            if os.path.isdir(clean_p):
                for root, _, files in os.walk(clean_p):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in SUPPORTED_EXTENSIONS:
                            paths.append(os.path.join(root, file))
            elif os.path.isfile(clean_p):
                paths.append(clean_p)
            else:
                matches = glob.glob(clean_p)
                if matches:
                    paths.extend([m for m in matches if os.path.splitext(m)[1].lower() in SUPPORTED_EXTENSIONS])
                else:
                    console.print(f"[bold yellow]Warning:[/] File '{clean_p}' not found.")

    # Default fallback to image.png if present and no interactive flag
    if not paths and not args.interactive and os.path.isfile("image.png"):
        paths.append("image.png")

    return sorted(list(dict.fromkeys(paths)))


def run_interactive_mode(detector: AIImageDetector, verbose: bool = False) -> None:
    """Friendly interactive drag-and-drop prompt."""
    render_banner()
    console.print("\n[bold cyan]Interactive Mode:[/] Drag & drop an image file into this window and hit [bold]Enter[/].")
    console.print("[dim]Type 'q' or 'exit' to quit.[/]\n")

    while True:
        try:
            raw_input = input("[Image file] > ").strip()
            if not raw_input:
                continue
            if raw_input.lower() in ["q", "exit", "quit"]:
                console.print("[dim]Goodbye![/]")
                break

            clean_path = raw_input.strip('"\'& ')
            if not os.path.isfile(clean_path):
                console.print(f"[bold red]Error:[/] File not found: '{clean_path}'\n")
                continue

            with console.status(f"[bold cyan]Scanning {os.path.basename(clean_path)} across 4 layers...[/]", spinner="dots"):
                report = detector.analyze(clean_path)

            render_report(report, verbose=verbose)
            console.print()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting...[/]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}\n")


def render_batch_summary(reports: list) -> None:
    """Renders a clean summary table when scanning multiple images."""
    render_banner()
    table = Table(
        title="[bold]BATCH SCAN RESULTS[/]",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("File", style="bright_white", width=22)
    table.add_column("Dimensions", style="dim", width=13)
    table.add_column("Verdict", width=24)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("Detected Tool / Signal", style="cyan")

    for rep in reports:
        v = rep.verdict
        if "CONFIRMED" in v.value or "HIGH" in v.value:
            v_badge = "[bold red]AI GENERATED[/]"
        elif "SUSPICIOUS" in v.value:
            v_badge = "[bold yellow]SUSPICIOUS[/]"
        elif "AUTHENTIC" in v.value:
            v_badge = "[bold green]AUTHENTIC[/]"
        else:
            v_badge = "[dim]INCONCLUSIVE[/]"

        tool_or_sig = rep.detected_generator or rep.primary_indicator
        table.add_row(
            rep.file_info.file_name,
            rep.file_info.dimensions,
            v_badge,
            f"{rep.confidence:.1f}%",
            tool_or_sig[:30] + ("..." if len(tool_or_sig) > 30 else ""),
        )

    console.print(table)


def main() -> None:
    args = parse_arguments()
    detector = AIImageDetector(confidence_threshold=args.threshold)

    # Interactive mode trigger
    if args.interactive:
        run_interactive_mode(detector, verbose=args.verbose)
        return

    image_paths = collect_image_paths(args)

    if not image_paths:
        # If no arguments and no image.png, launch interactive mode automatically!
        run_interactive_mode(detector, verbose=args.verbose)
        return

    reports = []

    for path in image_paths:
        try:
            if not args.json_output and len(image_paths) == 1:
                with console.status(f"[bold cyan]Scanning {os.path.basename(path)} across 4 layers...[/]", spinner="dots"):
                    report = detector.analyze(path)
            else:
                report = detector.analyze(path)

            reports.append(report)

            # If user requested visual forensics map
            if args.save_forensics:
                out_path = args.save_forensics
                if len(image_paths) > 1:
                    base, ext = os.path.splitext(out_path)
                    out_path = f"{base}_{report.file_info.file_name}{ext}"
                success = ForensicVisualizer.generate_diagnostic_map(path, out_path)
                if success and not args.json_output:
                    console.print(f"[bold green][OK] Forensic visualizer map saved:[/] [cyan]{out_path}[/]")

            # If user requested HTML report
            if args.html:
                out_html = args.html
                if len(image_paths) > 1:
                    base, ext = os.path.splitext(out_html)
                    out_html = f"{base}_{report.file_info.file_name}{ext}"
                ok_html = HTMLReporter.generate(report, out_html)
                if ok_html and not args.json_output:
                    console.print(f"[bold green][OK] Interactive HTML report generated:[/] [cyan]{out_html}[/]")

            # Render single image dashboard
            if not args.json_output and len(image_paths) == 1:
                render_report(report, verbose=args.verbose)

        except Exception as e:
            console.print(f"[bold red]Error analyzing '{path}':[/] {e}")

    # Render batch summary if multiple files scanned
    if not args.json_output and len(image_paths) > 1:
        render_batch_summary(reports)

    # JSON export / output
    all_data = [r.to_dict() for r in reports]
    if args.json_output:
        print(json.dumps(all_data if len(all_data) > 1 else all_data[0], indent=2))

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_data if len(all_data) > 1 else all_data[0], f, indent=2)
            if not args.json_output:
                console.print(f"\n[bold green][OK] JSON report saved to:[/] [cyan]{args.output}[/]")
        except Exception as e:
            console.print(f"[bold red]Failed to save JSON report to '{args.output}':[/] {e}")


if __name__ == "__main__":
    main()