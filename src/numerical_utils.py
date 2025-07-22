"""
numerical_utils.py  ░  v0.1
--------------------------------
Numerical analysis utilities for curve tracing:
• Gradient estimation via finite differences
• Curvature computation via implicit derivatives and geometric methods
• Adaptive step size control based on curvature
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


def implicit_curvature(f: Callable, point: np.ndarray, gfun: Callable,
                       eval_counter: List[int], tolerance: float = 1e-6) -> float:
    """Compute curvature of implicit curve f(x,y)=0 at point using derivatives.

    Args:
        f: Implicit function f(x,y) = 0
        point: Point at which to compute curvature
        gfun: Gradient function
        eval_counter: List to track function evaluations
        tolerance: Finite difference tolerance

    Returns:
        Curvature value (positive scalar)
    """
    x, y = point
    finite_diff_step = tolerance * max(1.0, norm(point))

    # First derivatives (already computed)
    fx, fy = gfun(point)

    # Second derivatives via finite differences
    fxx = (f(np.array([x + finite_diff_step, y])) - 2 * f(point) +
           f(np.array([x - finite_diff_step, y]))) / finite_diff_step**2

    fyy = (f(np.array([x, y + finite_diff_step])) - 2 * f(point) +
           f(np.array([x, y - finite_diff_step]))) / finite_diff_step**2

    fxy = (f(np.array([x + finite_diff_step, y + finite_diff_step])) -
           f(np.array([x + finite_diff_step, y - finite_diff_step])) -
           f(np.array([x - finite_diff_step, y + finite_diff_step])) +
           f(np.array([x - finite_diff_step, y - finite_diff_step]))) / (4 * finite_diff_step**2)

    # Update function evaluation counter
    eval_counter[0] += 5  # 5 extra evaluations for second derivatives

    # Curvature formula
    grad_norm_sq = fx**2 + fy**2
    if grad_norm_sq < 1e-12:
        return 0.0

    numerator = abs(fxx * fy**2 - 2 * fxy * fx * fy + fyy * fx**2)
    denominator = grad_norm_sq**(3/2)

    return numerator / denominator


def compute_curvature_from_points(points: np.ndarray) -> np.ndarray:
    """Compute 3-point curvature values from a sequence of points.

    Args:
        points: Array of points [(x1,y1), (x2,y2), ...]

    Returns:
        Array of curvature values (NaN for first two points)
    """
    curvature_vals = np.zeros(len(points))
    curvature_vals[:2] = np.nan

    for i in range(2, len(points)):
        a, b, c = points[i-2], points[i-1], points[i]
        ab, bc, ac = b - a, c - b, c - a
        ab_norm, bc_norm, ac_norm = norm(ab), norm(bc), norm(ac)

        if ab_norm < 1e-12 or bc_norm < 1e-12 or ac_norm < 1e-12:
            curvature_vals[i] = np.nan
        else:
            cos_theta = np.clip((ab_norm**2 + bc_norm**2 - ac_norm**2) /
                               (2 * ab_norm * bc_norm), -1, 1)
            sin_theta = np.sqrt(1 - cos_theta**2)
            curvature_vals[i] = 2 * sin_theta / ac_norm

    return curvature_vals


def get_curvature(f: Callable, gfun: Callable, current_pos: np.ndarray,
                  pts: List[np.ndarray], eval_counter: List[int]) -> float:
    """Get curvature using appropriate method based on available points.

    Args:
        f: Implicit function
        gfun: Gradient function
        current_pos: Current position
        pts: List of traced points
        eval_counter: Function evaluation counter

    Returns:
        Curvature value
    """
    if len(pts) < 3:
        # Use implicit curvature for first two steps
        return implicit_curvature(f, current_pos, gfun, eval_counter)
    else:
        # Use efficient 3-point method
        return abs(three_pt_kappa(pts[-3], pts[-2], pts[-1]))


def next_step_size(step_size_prev: float, curvature_prev: float, curvature_current: float,
                   delta_prev: float, radius_prev: float, cfg) -> tuple[float, float]:
    """Compute next step size using geometric mean trust-radius method.

    Args:
        step_size_prev: Previous step size
        curvature_prev: Previous curvature
        curvature_current: Current curvature
        delta_prev: Previous Newton displacement
        radius_prev: Previous trust region radius
        cfg: TraceConfig object with step size bounds

    Returns:
        Tuple of (new_step_size, new_radius)
    """
    curvature_current = max(curvature_current, 1e-12)

    # Geometric mean update: smooth log-space evolution
    radius = np.sqrt(radius_prev * max(delta_prev, cfg.rho_min))

    # Standard predictor step size
    step_size = (2.0 * radius / curvature_current)**0.5
    step_size_clipped = np.clip(step_size, cfg.ds_min, cfg.ds_max)

    return step_size_clipped, radius


def initialize_step_control(f: Callable, gfun: Callable, p0: np.ndarray,
                          eval_counter: List[int], cfg) -> tuple[float, float]:
    """Initialize step size using implicit curvature at the starting point.

    Args:
        f: Implicit function
        gfun: Gradient function
        p0: Starting point
        eval_counter: Function evaluation counter
        cfg: TraceConfig object

    Returns:
        Tuple of (initial_step_size, initial_curvature)
    """
    initial_curvature = implicit_curvature(f, p0, gfun, eval_counter)
    # Use ds_max to allow the boldest safe start; formula only constrains for high curvature
    initial_step_size = np.clip((2.0 * cfg.ds_max * max(1.0, norm(p0)) /
                                max(initial_curvature, 1e-12))**0.5, cfg.ds_min, cfg.ds_max)

    return initial_step_size, initial_curvature