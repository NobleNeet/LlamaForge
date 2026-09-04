#!/usr/bin/env python3
"""Compatibility wrapper for the collector entrypoint."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).with_name("llamaforge_metrics.py")
    runpy.run_path(str(target), run_name="__main__")
