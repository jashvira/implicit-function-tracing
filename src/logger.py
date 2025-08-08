"""
Enhanced logging system for curve tracing with structured output and visual formatting.
"""
import logging
import sys
from datetime import datetime
from typing import Optional

class TraceLogger:
    """Enhanced logger for curve tracing with structured output."""

    def __init__(self):
        self.enabled = False
        self.logger = logging.getLogger('tracer')
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatter with structured output
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # File handler for detailed logs (always enabled)
        file_handler = logging.FileHandler('trace.log', mode='w')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Track step information
        self.current_step = 0
        self.newton_attempts = 0
        self.console_enabled = False

    def enable(self, console_output: bool = False):
        """Enable logging output."""
        self.enabled = True
        self.console_enabled = console_output

        # Add console handler only if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.info("=" * 60)
        self.logger.info("CURVE TRACING SESSION STARTED")
        self.logger.info("=" * 60)

    def log(self, message: str, level: str = "INFO"):
        """Log a message with structured formatting."""
        if not self.enabled:
            return

        # Add step context if available
        if self.current_step > 0:
            message = f"[Step {self.current_step}] {message}"

        if level == "SUCCESS":
            self.logger.info(f"[SUCCESS] {message}")
        elif level == "WARN":
            self.logger.warning(f"[WARN] {message}")
        elif level == "ERROR":
            self.logger.error(f"[ERROR] {message}")
        else:
            self.logger.info(f"[INFO] {message}")

    def start_step(self, step_num: int, position: str):
        """Log the start of a new tracing step."""
        self.current_step = step_num
        self.newton_attempts = 0
        self.logger.info("")
        self.logger.info(f"STEP {step_num:3d} | Position: {position}")
        self.logger.info("-" * 40)

    def newton_attempt(self, attempt_num: int, radius: float, start_pos: str):
        """Log a Newton attempt."""
        self.newton_attempts = attempt_num
        self.logger.info(f"Newton Attempt {attempt_num} | Radius: {radius:.3e} | Start: {start_pos}")

    def newton_iteration(self, iter_num: int, position: str, fval: float, grad_norm: float):
        """Log a Newton iteration."""
        indent = "  " * (self.newton_attempts + 1)
        self.logger.info(f"{indent}Iter {iter_num:2d}: pos={position} | f={fval:.3e} | |∇f|={grad_norm:.3e}")

    def newton_step(self, delta: str, delta_norm: float, clipped: bool):
        """Log a Newton step."""
        indent = "  " * (self.newton_attempts + 1)
        clip_indicator = " [CLIPPED]" if clipped else ""
        self.logger.info(f"{indent}    δ={delta} | |δ|={delta_norm:.3e}{clip_indicator}")

    def newton_result(self, success: bool, final_pos: str, iterations: int, reason: str = ""):
        """Log Newton result."""
        status = "CONVERGED" if success else "FAILED"
        self.logger.info(f"Newton {status} in {iterations} iterations | Final: {final_pos}")
        if reason:
            self.logger.info(f"   Reason: {reason}")

    def step_result(self, success: bool, inside: bool, position: str):
        """Log step result."""
        if success and inside:
            self.logger.info(f"STEP SUCCESS | Position: {position}")
        else:
            status = "STEP FAILED"
            if not success:
                status += " (Newton failed)"
            if not inside:
                status += " (Outside box)"
            self.logger.warning(f"{status} | Position: {position}")

    def step_size_update(self, old_size: float, new_size: float, metric: float, radius: float, metric_name: str = "CurvVar"):
        """Log step size update with a named metric (e.g., curvature variation)."""
        arrow = "↓" if new_size < old_size else "↑" if new_size > old_size else "="
        self.logger.info(f"Step size: {old_size:.3e} {arrow} {new_size:.3e}")
        self.logger.info(f"   {metric_name}: {metric:.3f} | Radius: {radius:.3e}")

    def endpoint_snap(self, target: str):
        """Log endpoint snapping."""
        self.logger.info("SNAPPED TO ENDPOINT")
        self.logger.info(f"   Target: {target}")

    def trace_complete(self, reason: str, points: int, evals: int):
        """Log trace completion."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"TRACE COMPLETED: {reason.upper()}")
        self.logger.info(f"   Points: {points} | Function evaluations: {evals}")
        self.logger.info("=" * 60)

# Global logger instance
logger = TraceLogger()