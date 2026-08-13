"""Standalone, research-grade OpenVLA + LIBERO data collection.

This package intentionally does not change the original Event-SAE collection
path.  It records richer simulator and policy telemetry while keeping the
original layer-31 dense activation shard format for downstream SAE training.
"""

from .config import ExtendedRunConfig, load_extended_config, parse_overrides

__all__ = ["ExtendedRunConfig", "load_extended_config", "parse_overrides"]
