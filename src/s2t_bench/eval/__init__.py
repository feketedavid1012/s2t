"""Evaluation harnesses: Whisper (accuracy/speed/hardware) and Gemma (correction
+ JSON schema conformance), each under simulated concurrency."""
from .concurrency import ConcurrencyResult, run_concurrent
from .hardware import HardwareMonitor

__all__ = ["ConcurrencyResult", "run_concurrent", "HardwareMonitor"]
