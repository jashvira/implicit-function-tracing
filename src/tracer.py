"""
box_curve_tracer.py  ░  v0.2
--------------------------------
Curve–tracing utilities when the curve is *known* to lie INSIDE a
rectangular frame and both end‑points lie **on** that frame.

Key take‑aways folded in from the user's experimental script:
• inward‑tangent selection using box topology (no look‑ahead needed)
• trust‑region Newton with dual tolerance system (function + step size)
• intelligent endpoint snapping for boundary precision handling
• arc‑length + distance termination (robust for any winding)
• curvature‑adaptive step control (keeps points sparse in flats)
• enhanced boundary checking with natural operators

v0.2 improvements:
• Eliminated duplicate Newton calls for better performance
• Added intelligent endpoint snapping to handle boundary precision issues
• Dual tolerance system prevents unnecessary Newton iterations
• Simplified boundary logic with configurable snap_eps parameter
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass
from typing import NamedTuple

# Import logger - handle both relative and absolute imports
try:
    from .logger import logger
except ImportError:
    from logger import logger

# Import numerical utilities
try:
    from .numerical_utils import (
        get_curvature, next_step_size, initialize_step_control
    )
except ImportError:
    from numerical_utils import (
        get_curvature, next_step_size, initialize_step_control
    )

# ---------------------------------------------------------------------------
# ### 0. UTILITY HELPERS
# ---------------------------------------------------------------------------


def plot_delta_distribution(delta_history, title="Delta Distribution Over Time"):
    """Plot histogram and time series of Newton delta values."""
    import matplotlib.pyplot as plt

    if not delta_history:
        print("No delta history to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Time series
    ax1.plot(delta_history, 'b-', alpha=0.7)
    ax1.set_xlabel('Newton Iteration')
    ax1.set_ylabel('Delta Magnitude')
    ax1.set_title('Delta Over Time')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Histogram
    ax2.hist(delta_history, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Delta Magnitude')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Delta Distribution')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

# ---------------------------------------------------------------------------
# ### 1. BOX TOPOLOGY HELPERS
# ---------------------------------------------------------------------------


def classify_edges(point: np.ndarray, box_min: np.ndarray, box_max: np.ndarray, tol=1e-9) -> set[str]:
    x, y = point
    xmin, ymin = box_min
    xmax, ymax = box_max
    edges = set()
    if abs(x - xmin) < tol:
        edges.add("left")
    if abs(x - xmax) < tol:
        edges.add("right")
    if abs(y - ymin) < tol:
        edges.add("bottom")
    if abs(y - ymax) < tol:
        edges.add("top")
    return edges


def inward_tangent(tangent_vec: np.ndarray, p0: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Flip *tangent_vec* if its first micro‑step exits the box (ensures inward march)."""
    trial = p0 + 1e-3 * tangent_vec
    xmin, ymin = box_min
    xmax, ymax = box_max
    inside = xmin < trial[0] < xmax and ymin < trial[1] < ymax
    return tangent_vec if inside else -tangent_vec

# ---------------------------------------------------------------------------
# ### 2. TRUST‑REGION NEWTON SOLVER
# ---------------------------------------------------------------------------


def newton_tr(f, gfun, x_pred, prev_tangent_vec, *, step0, max_iter=10, max_shrink=5,
              tol: float = 1e-5, dn_tol: float = 1e-3, grad_tol: float = 1e-6, shrink: float = 0.5, counter=None, delta_history=None, attempts=None):
    """Robust Newton that always returns (x, newton_success, final_radius).

    If attempts is provided, logs each Newton iteration as:
    (iteration_idx, position, function_value, clipped, converged)
    """
    radius = step0

    for shrink_iter in range(max_shrink):
        logger.newton_attempt(shrink_iter+1, radius, f"[{x_pred[0]:.6f}, {x_pred[1]:.6f}]")
        x = x_pred.copy()

        # Track cumulative displacement from x_pred
        cumulative_displacement = 0.0

        for newton_iter in range(max_iter):
            fval = f(x)
            if counter is not None: counter[0] += 1
            gradient_vec = gfun(x)

            logger.newton_iteration(newton_iter+1, f"[{x[0]:.6f}, {x[1]:.6f}]", fval, norm(gradient_vec))

            if norm(gradient_vec) < grad_tol:
                logger.log(f"Gradient too small |∇f|={norm(gradient_vec):.3e}", "WARN")
                return x_pred, False, radius

            # Full 2D Newton with prediction constraint
            constraint_val = (x - x_pred) @ prev_tangent_vec
            F = np.array([fval, constraint_val])
            J = np.vstack((gradient_vec, prev_tangent_vec))

            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError as e:
                logger.log("Singular jacobian", "ERROR")
                break

            delta_norm = norm(delta)
            remaining_radius = radius - cumulative_displacement

            clipped = False
            if delta_norm > remaining_radius:
                delta *= remaining_radius / delta_norm
                delta_norm = remaining_radius
                clipped = True
                logger.log(f"CLIPPED: |δ|={delta_norm:.3e} (remaining budget: {remaining_radius:.3e})", "WARN")

            # Log this attempt if requested
            if attempts is not None:
                attempts.append((newton_iter, x.copy(), fval, clipped, False))

            # Store the actual used delta (after clipping)
            if delta_history is not None:
                delta_history.append(delta_norm)

            x += delta
            cumulative_displacement += delta_norm

            logger.newton_step(f"[{delta[0]:.6f}, {delta[1]:.6f}]", delta_norm, clipped)

            # Early shrink on rim-hit: if we're at the rim and still not converged, shrink immediately
            if clipped and abs(fval) > tol and remaining_radius < 1e-12:
                logger.log(f"RIM-HIT: |f|={abs(fval):.3e} > tol, shrinking immediately", "WARN")
                break

            if delta_norm < dn_tol and abs(fval) < tol:
                logger.newton_result(True, f"[{x[0]:.6f}, {x[1]:.6f}]", newton_iter+1)
                # Log final converged attempt
                if attempts is not None:
                    new_fval = f(x)
                    if counter is not None: counter[0] += 1
                    attempts.append((newton_iter, x.copy(), new_fval, clipped, True))
                return x, True, radius

        logger.log("Newton failed to converge; shrinking radius", "WARN")
        radius *= shrink

    logger.log("Newton exhausted all trust region attempts", "ERROR")
    return x_pred, False, radius

# ---------------------------------------------------------------------------
# ### 3. CURVATURE & STEP CONTROL HELPERS
# ---------------------------------------------------------------------------


def _check_endpoint_snap(current_pos: np.ndarray, p1: np.ndarray, snap_eps: float, pts: list, eval_counter: list, delta_history: list, radius_history: list, kappa_history: list, step_size_history: list, captured_steps: list) -> TraceResult:
    """Check if we should snap to endpoint and return TraceResult if so."""
    if norm(current_pos - p1) < snap_eps:
        pts.append(p1.copy())
        # Maintain radius_history length alignment by duplicating last radius
        if radius_history is not None and len(radius_history) > 0:
            radius_history.append(radius_history[-1])
        # Maintain kappa_history length alignment by duplicating last kappa
        if kappa_history is not None and len(kappa_history) > 0:
            kappa_history.append(kappa_history[-1])
        # Maintain step_size_history length alignment by duplicating last step size
        if step_size_history is not None and len(step_size_history) > 0:
            step_size_history.append(step_size_history[-1])
        logger.log("Snapped to endpoint", "SUCCESS")
        return TraceResult(np.asarray(pts), True, "snapped_to_endpoint", eval_counter[0], delta_history, radius_history, kappa_history, step_size_history, captured_steps)
    return None







# ---------------------------------------------------------------------------
# ### 4. MAIN TRACING FUNCTION
# ---------------------------------------------------------------------------


@dataclass
class TraceConfig:
    ds_min: float = 5e-3 # keep this small enough for snapping to work! ! Should be set using GC
    ds_max: float = 2e-1 # This massively controls the crazy newton behaviour, i.e. great for keeping it on leash. ! Should be set using box size
    arc_eps: float = 1e-5
    snap_eps: float = 1e-3
    max_iter: int = 1000
    rho_min: float = 1e-12
    shrink: float = 0.9 # This guy gets things moving off the high curvature spots. The bigger, more liberal it is.
    max_shrink: int = 1 # This guy just doubles down on the shrink effect


class TraceResult(NamedTuple):
    points: np.ndarray
    completed: bool
    reason: str
    f_evals: int
    delta_history: list = None
    radius_history: list = None
    kappa_history: list = None
    step_size_history: list = None
    captured_steps: list = None


@dataclass
class StepState:
    """State captured for one tracing step."""
    step_index: int
    current_position: np.ndarray
    predicted_position: np.ndarray
    trust_radius: float
    newton_attempts: list  # [(iter, pos, fval, clipped, converged), ...]
    final_position: np.ndarray
    newton_succeeded: bool
    step_succeeded: bool
    reason: str


def trace_curve_in_box(p0: np.ndarray, p1: np.ndarray, f, box_min, box_max, cfg: TraceConfig = TraceConfig(), track_deltas: bool = False, track_radii: bool = False, track_kappa: bool = False, track_step_sizes: bool = False, capture_steps: bool = False, enable_logging: bool = False, console_output: bool = False, gfun_override=None) -> TraceResult:
    """Trace the f=0 curve inside `box` from p0 to p1 (both on frame)."""
    if enable_logging:
        logger.enable(console_output=console_output)

    logger.log(f"Starting trace from [{p0[0]:.6f}, {p0[1]:.6f}] to [{p1[0]:.6f}, {p1[1]:.6f}]")

    # ### INITIALIZATION
    delta_history = [] if track_deltas else None
    radius_history = [] if track_radii else None
    kappa_history = [] if track_kappa else None
    step_size_history = [] if track_step_sizes else None
    captured_steps = [] if capture_steps else None

    eval_counter = [0]
    # Caller MUST provide a gradient function (e.g. from a neural network)
    if gfun_override is None:
        raise ValueError("trace_curve_in_box now expects gfun_override (gradient function) to be provided.")

    def gfun(q):
        return gfun_override(q, eval_counter)

    # ### SETUP INITIAL TANGENT
    gradient_vec_initial = gfun(p0)
    tangent_vec = np.array([-gradient_vec_initial[1], gradient_vec_initial[0]])
    tangent_vec /= norm(tangent_vec)
    tangent_vec = inward_tangent(tangent_vec, p0, box_min, box_max)

    pts = [p0.copy()]
    current_pos = p0.copy()

    # ### INITIALIZE STEP CONTROL
    step_size, curvature_prev = initialize_step_control(f, gfun, p0, eval_counter, cfg)
    logger.log(f"Initialization: curvature_0={curvature_prev:.3f}, step_size_0={step_size:.3e} (implicit)")

    def inside(q): return (box_min[0] <= q[0] <= box_max[0] and
                           box_min[1] <= q[1] <= box_max[1])

    # ### MAIN TRACING LOOP
    for step in range(cfg.max_iter):
        logger.start_step(step, f"[{current_pos[0]:.6f}, {current_pos[1]:.6f}]")

        # ### PREDICT NEXT POSITION
        predicted_pos = current_pos + step_size*tangent_vec
        predicted_pos = np.clip(predicted_pos, box_min, box_max)

        logger.log(f"Newton call: step={step}, ds={step_size:.3e}")

        # ### NEWTON CORRECTION
        # enforce a minimal trust-region radius so Newton never stalls in place
        init_rad = max(step_size, cfg.rho_min)

        # Capture Newton attempts if requested
        attempts = [] if capture_steps else None

        corrected_pos, newton_success, final_radius = newton_tr(f, gfun, predicted_pos, tangent_vec, step0=init_rad,
                                    max_shrink=cfg.max_shrink, shrink=cfg.shrink, counter=eval_counter, delta_history=delta_history, attempts=attempts)

        step_success = newton_success and inside(corrected_pos)

        logger.step_result(newton_success, inside(corrected_pos), f"[{corrected_pos[0]:.6f}, {corrected_pos[1]:.6f}]")

        # ### CAPTURE STEP STATE (if requested)
        if capture_steps:
            reason = "newton_ok" if newton_success else "newton_failed"
            if newton_success and not inside(corrected_pos):
                reason = "outside_box"

            step_state = StepState(
                step_index=step,
                current_position=current_pos.copy(),
                predicted_position=predicted_pos.copy(),
                trust_radius=init_rad,
                newton_attempts=attempts,
                final_position=corrected_pos.copy(),
                newton_succeeded=newton_success,
                step_succeeded=step_success,
                reason=reason
            )
            captured_steps.append(step_state)

        # ### HANDLE STEP FAILURE
        if not newton_success or not inside(corrected_pos):
            # ### ENDPOINT SNAPPING (if close)
            snap_result = _check_endpoint_snap(current_pos, p1, cfg.snap_eps, pts, eval_counter, delta_history, radius_history, kappa_history, step_size_history, captured_steps)
            if snap_result is not None:
                logger.endpoint_snap(f"[{p1[0]:.6f}, {p1[1]:.6f}]")
                logger.trace_complete("snapped_to_endpoint", len(pts), eval_counter[0])
                return TraceResult(np.asarray(pts), True, "snapped_to_endpoint", eval_counter[0], delta_history, radius_history, kappa_history, step_size_history, captured_steps)

            # ### STEP SIZE REDUCTION (failure recovery)
            step_size_old = step_size
            step_size = max(final_radius*cfg.shrink, cfg.ds_min)
            logger.log(f"Step size reduction: {step_size_old:.3e} → {step_size:.3e} (failure)")
            if step_size <= cfg.ds_min:
                logger.log("Aborting: minimum step size reached", "ERROR")
                logger.trace_complete("minimum_step_size", len(pts), eval_counter[0])
                return TraceResult(np.asarray(pts), False, "minimum_step_size", eval_counter[0], delta_history, radius_history, kappa_history, step_size_history, captured_steps)
            continue

        # ### PROCESS SUCCESSFUL STEP
        pts.append(corrected_pos.copy())
        current_pos = corrected_pos

        # ### RECORD HISTORIES (if requested)
        # Record the Newton trust-region radius for this accepted point
        if radius_history is not None:
            radius_history.append(final_radius)

        # Record the curvature for this accepted point
        if kappa_history is not None:
            kappa_history.append(curvature_prev)  # Use the curvature from previous step that led to this point

        # Record the step size used for this step
        if step_size_history is not None:
            step_size_history.append(step_size)

        # ### CHECK TERMINATION
        dist_to_target = norm(current_pos - p1)
        if dist_to_target < cfg.arc_eps:
            logger.log("Reached target successfully", "SUCCESS")
            logger.trace_complete("reached_target", len(pts), eval_counter[0])
            return TraceResult(np.asarray(pts), True, "reached_target", eval_counter[0], delta_history, radius_history, kappa_history, step_size_history, captured_steps)

        # ### UPDATE STEP CONTROL
        curvature_current = get_curvature(f, gfun, current_pos, pts, eval_counter)
        delta_prev = norm(corrected_pos - predicted_pos)  # Track actual Newton pull-back
        step_size_new, current_radius = next_step_size(step_size, curvature_prev, curvature_current, delta_prev, final_radius, cfg)
        curvature_prev = curvature_current

        logger.step_size_update(step_size, step_size_new, curvature_current, current_radius)

        step_size = step_size_new

        # ### UPDATE TANGENT DIRECTION
        gradient_vec = gfun(current_pos)
        if norm(gradient_vec) < 1e-9:
            logger.log("Gradient too small, continuing", "WARN")
            continue
        candidate_tangent_vec = np.array([-gradient_vec[1], gradient_vec[0]])
        if candidate_tangent_vec @ tangent_vec < 0:
            candidate_tangent_vec = -candidate_tangent_vec
        tangent_vec = candidate_tangent_vec / norm(candidate_tangent_vec)
