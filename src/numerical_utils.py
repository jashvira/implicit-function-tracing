"""
numerical_utils.py  ░  v0.2
--------------------------------
Numerical analysis utilities for curve tracing:
• Gradient estimation via finite differences
• 3-point curvature and curvature-variation helper
• κ-free adaptive step size control driven by trust radius and curvature variation
• Trust region management for Newton iterations

This module provides the core numerical analysis functions used by the curve tracer,
abstracted from the main tracing logic for better modularity and reusability.
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from typing import Callable, Optional, List


def adaptive_grad(f: Callable, point: np.ndarray, eps_scale: float = 1e-6,
                  counter: Optional[List[int]] = None) -> np.ndarray:
    """Adaptive central‑difference gradient (O(‖point‖‑scaled ε)).

    Args:
        f: Function to differentiate
        point: Point at which to compute gradient
        eps_scale: Scale factor for adaptive epsilon
        counter: Optional list to track function evaluations

    Returns:
        Gradient vector [∂f/∂x, ∂f/∂y]
    """
    epsilon = eps_scale * max(1.0, norm(point))
    gradient_vec = np.empty(2)
    f0 = f(point)
    if counter is not None:
        counter[0] += 1

    for i in range(2):
        delta_point = np.zeros(2)
        delta_point[i] = epsilon
        gradient_vec[i] = (f(point + delta_point) - f0) / epsilon
        if counter is not None:
            counter[0] += 1

    return gradient_vec


def three_pt_kappa(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray,
                   tolerance: float = 1e-12) -> float:
    """Compute curvature using three-point geometric method.

    Args:
        point_a: First point
        point_b: Second point
        point_c: Third point
        tolerance: Minimum distance tolerance

    Returns:
        Curvature value (positive scalar)
    """
    ab, bc, ac = point_b - point_a, point_c - point_b, point_c - point_a
    abn, bcn, acn = norm(ab), norm(bc), norm(ac)

    if abn < tolerance or bcn < tolerance or acn < tolerance:
        return 0.0

    cos_t = np.clip((abn**2 + bcn**2 - acn**2) / (2 * abn * bcn), -1, 1)
    sin_t = np.sqrt(1 - cos_t**2)
    return 2 * sin_t / acn


def compute_hessian_norm_jax(f: Callable, point: np.ndarray) -> float:
    """Compute Frobenius norm of Hessian using JAX autodiff.

    Expects f to accept a numpy/jax array of shape (2,) and return a scalar compatible with JAX tracing.
    """
    import jax
    import jax.numpy as jnp

    def f_wrapped(x_j):
        # Expect f to handle JAX arrays and return a JAX scalar
        return jnp.asarray(f(x_j))

    H = jax.hessian(f_wrapped)(jnp.asarray(point))
    return float(jnp.linalg.norm(H, ord='fro'))


def curvvar_bootstrap_from_hessian(
    prev_hess_norm: Optional[float],
    curr_hess_norm: float,
    eps: float = 1e-4,
    clip: Optional[float] = None,
) -> Optional[float]:
    """Relative change of Hessian norm as a bootstrap CurvVar proxy.

    Uses a larger epsilon and optional clipping to avoid extreme spikes
    during the first few steps when the Hessian norm can be tiny.

    Returns None if prev_hess_norm is not available yet.
    """
    if prev_hess_norm is None:
        return None
    val = abs(curr_hess_norm - prev_hess_norm) / max(curr_hess_norm, eps)
    if clip is not None:
        val = min(val, clip)
    return val


def curvature_variation_from_points(pts: List[np.ndarray], eps: float = 1e-12) -> float:
    """Estimate relative curvature change using last four points.

    Uses two consecutive 3-point curvature estimates based on the last four
    accepted points: k1 from (p_{-4}, p_{-3}, p_{-2}) and k2 from
    (p_{-3}, p_{-2}, p_{-1}). Returns a relative change metric
    r_k = |k2 - k1| / max(k2, eps).

    Args:
        pts: Sequence of accepted points; must have length >= 4.
        eps: Small positive number to avoid division by zero.

    Returns:
        Non-negative scalar measuring curvature variation. If insufficient
        points are available, returns 0.0.
    """
    if len(pts) < 4:
        return 0.0

    k1 = abs(three_pt_kappa(pts[-4], pts[-3], pts[-2]))
    k2 = abs(three_pt_kappa(pts[-3], pts[-2], pts[-1]))

    return abs(k2 - k1) / max(k2, eps)


def initialize_step_control_kfree(p0: np.ndarray, cfg) -> float:
    """Initialize step size without curvature, using a conservative fraction of ds_max.

    Args:
        p0: Starting point (unused, reserved for future scaling)
        cfg: TraceConfig object

    Returns:
        Initial step size.
    """
    # Conservative but configurable start; avoids big leaps before radius stabilizes
    init_frac = getattr(cfg, 'init_step_fraction', 0.5)
    return np.clip(init_frac * cfg.ds_max, cfg.ds_min, cfg.ds_max)


def next_step_size_kfree(delta_prev: float, radius_prev: float, cfg,
                         curv_var: Optional[float] = None, alpha: Optional[float] = 0.5,
                         power: float = 0.5, beta: float = 1.0) -> tuple[float, float]:
    """Curvature-free step-size update driven by trust radius and curvature variation.

    Args:
        delta_prev: Previous Newton displacement (pull-back)
        radius_prev: Previous trust region radius
        cfg: TraceConfig with bounds; rho_min is derived internally from ds_min
        curv_var: Relative curvature change r_k (>=0) from last 4 points
        alpha: Damping gain (>0)
        power: Exponent for damping (0.5 recommended)
        beta: Gain from radius to step size (1.0 recommended)

    Returns:
        (new_step_size, new_radius)
    """
    # Derive a conservative lower bound for the trust-region radius from ds_min
    rho_factor = getattr(cfg, 'rho_min_factor', 0.25)
    rho_min = rho_factor * cfg.ds_min
    # Update trust radius via geometric mean of last radius and pull-back
    radius = np.sqrt(max(radius_prev, rho_min) * max(delta_prev, rho_min))

    # Base predictor from radius only
    step_size = beta * radius

    # Apply curvature-variation damping
    if curv_var is not None and alpha is not None and alpha > 0:
        denom = (1.0 + alpha * max(curv_var, 0.0))**max(power, 0.0)
        # Optional cap on damping to avoid collapse; configured via TraceConfig
        cap = getattr(cfg, 'curvvar_damping_cap', float('inf'))
        if np.isfinite(cap):
            denom = min(denom, cap)
        step_size = step_size / denom

    step_size_clipped = np.clip(step_size, cfg.ds_min, cfg.ds_max)
    return step_size_clipped, radius