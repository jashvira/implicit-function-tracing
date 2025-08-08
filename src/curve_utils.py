import numpy as np
import jax.numpy as jnp

def get_curve_functions():
    """Return a dictionary of available curve functions."""
    return {
        'sine': lambda x: jnp.sin(2*jnp.pi*x),
        'double_sine': lambda x: jnp.sin(6*jnp.pi*x) + 0.5*jnp.sin(8*jnp.pi*x),
        'damped_sine': lambda x: jnp.exp(-2*x) * jnp.sin(6*jnp.pi*x),
        'sine_cos': lambda x: jnp.sin(3*jnp.pi*x) * jnp.cos(2*jnp.pi*x),
        'cubic': lambda x: 4*x**3 - 6*x**2 + 2*x,
        'heart': lambda x: 0.5*jnp.sin(jnp.pi*x) * (1 + 0.3*jnp.cos(4*jnp.pi*x)),
        'spiral': lambda x: 0.3*x*jnp.sin(8*jnp.pi*x),
        'bumpy': lambda x: jnp.sin(2*jnp.pi*x) + 0.3*jnp.sin(10*jnp.pi*x) + 0.1*jnp.sin(20*jnp.pi*x),
        'exponential': lambda x: jnp.exp(-3*x),
        'hyperbola': lambda x: 1/(x+0.2)
    }

def calculate_box_bounds(points, point1_edge, point2_edge, x_padding, y_padding):
    """Calculate box bounds based on points and edge assignments.

    Args:
        points (np.ndarray): Array of shape (2, 2) containing two points
        point1_edge (str): Edge for first point ('left', 'right', 'top', 'bottom')
        point2_edge (str): Edge for second point ('left', 'right', 'top', 'bottom')
        x_padding (float): Padding in x direction
        y_padding (float): Padding in y direction

    Returns:
        tuple: (box_min, box_max) as np.arrays
    """
    p1, p2 = points[0], points[1]

    # Initial bounds based on first point
    if point1_edge == 'left':
        x_min = p1[0]
        x_max = max(p2[0], p1[0]) + x_padding
    elif point1_edge == 'right':
        x_max = p1[0]
        x_min = min(p2[0], p1[0]) - x_padding
    else:  # top or bottom
        x_min = min(p1[0], p2[0]) - x_padding
        x_max = max(p1[0], p2[0]) + x_padding

    if point1_edge == 'top':
        y_max = p1[1]
        y_min = min(p2[1], p1[1]) - y_padding
    elif point1_edge == 'bottom':
        y_min = p1[1]
        y_max = max(p2[1], p1[1]) + y_padding
    else:  # left or right
        y_min = min(p1[1], p2[1]) - y_padding
        y_max = max(p1[1], p2[1]) + y_padding

    # Adjust bounds based on second point
    if point2_edge == 'left':
        x_min = p2[0]
    elif point2_edge == 'right':
        x_max = p2[0]
    elif point2_edge == 'top':
        y_max = p2[1]
    elif point2_edge == 'bottom':
        y_min = p2[1]

    return np.array([x_min, y_min]), np.array([x_max, y_max])

def _plot_curve_and_trace(ax, curve_func, box_min, box_max, p0f, p1f, results, curve_name):
    """Plot the main curve and trace on the given axis."""
    # Plot the theoretical curve
    x = np.linspace(box_min[0]-0.2, box_max[0]+0.2, 800)
    y_curve = curve_func(x)
    # Convert JAX array to numpy if needed
    if hasattr(y_curve, 'block_until_ready'):
        y_curve = np.asarray(y_curve)
    ax.plot(x, y_curve, 'k--', alpha=0.6, lw=1.5, label='curve')

    # Plot the bounding box
    ax.plot([box_min[0], box_max[0], box_max[0], box_min[0], box_min[0]],
            [box_min[1], box_min[1], box_max[1], box_max[1], box_min[1]],
            'gray', lw=2, alpha=0.7, label='box')

    # Plot start and end points
    ax.scatter([p0f[0]], [p0f[1]], c='green', s=100, zorder=5, label='start')
    ax.scatter([p1f[0]], [p1f[1]], c='red', s=100, zorder=5, label='end')

    # Plot the traced curve
    if results.completed:
        ax.plot(results.points[:,0], results.points[:,1], 'b-o', ms=3, lw=2, alpha=0.8, label='traced')
    else:
        ax.plot(results.points[:,0], results.points[:,1], 'b-o', ms=3, lw=2, alpha=0.8,
                label=f'partial ({results.reason})')

    ax.set_ylabel('y')
    ax.set_title(f'{curve_name.title()} Curve Trace | {len(results.points)} points')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)
    ax.grid(True, alpha=0.3)


def _plot_radius_evolution(ax, results):
    """Plot the trust-region radius evolution on the given axis."""
    if results.radius_history and len(results.radius_history) > 0:
        x_coords = results.points[:,0]
        radius_vals = results.radius_history

        # Handle different length scenarios
        if len(radius_vals) == len(x_coords):
            ax.plot(x_coords, radius_vals, 'g-o', alpha=0.8, linewidth=2, ms=2)
        elif len(radius_vals) == len(x_coords) - 1:
            ax.plot(x_coords[1:], radius_vals, 'g-o', alpha=0.8, linewidth=2, ms=2)
        else:
            ax.plot(range(len(radius_vals)), radius_vals, 'g-o', alpha=0.8, linewidth=2, ms=2)
            ax.text(0.5, 0.95, f'Warning: radius_history length {len(radius_vals)} does not match points {len(x_coords)}',
                    ha='center', va='top', transform=ax.transAxes, color='red', fontsize=10)

        # Add min/max reference lines
        min_radius = min(radius_vals)
        max_radius = max(radius_vals)
        ax.axhline(min_radius, color='red', linestyle=':', alpha=0.5, label=f'Min: {min_radius:.2e}')
        ax.axhline(max_radius, color='blue', linestyle=':', alpha=0.5, label=f'Max: {max_radius:.2e}')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)
    else:
        ax.text(0.5, 0.5, 'No radius history available\n(set track_radii=True)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_ylabel('Trust-Region Radius')
    ax.set_title('Radius Evolution Along Curve')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)


def _plot_step_size_evolution(ax, results):
    """Plot the step size evolution on the given axis."""
    if hasattr(results, 'step_size_history') and results.step_size_history and len(results.step_size_history) > 0:
        x_coords = results.points[:,0]
        step_size_vals = results.step_size_history

        # Handle different length scenarios
        if len(step_size_vals) == len(x_coords):
            ax.plot(x_coords, step_size_vals, 'c-o', alpha=0.8, linewidth=2, ms=2)
        elif len(step_size_vals) == len(x_coords) - 1:
            ax.plot(x_coords[1:], step_size_vals, 'c-o', alpha=0.8, linewidth=2, ms=2)
        else:
            ax.plot(range(len(step_size_vals)), step_size_vals, 'c-o', alpha=0.8, linewidth=2, ms=2)
            ax.text(0.5, 0.95, f'Warning: step_size_history length {len(step_size_vals)} does not match points {len(x_coords)}',
                    ha='center', va='top', transform=ax.transAxes, color='red', fontsize=10)

        # Add min/max reference lines
        min_step = min(step_size_vals)
        max_step = max(step_size_vals)
        ax.axhline(min_step, color='red', linestyle=':', alpha=0.5, label=f'Min: {min_step:.2e}')
        ax.axhline(max_step, color='blue', linestyle=':', alpha=0.5, label=f'Max: {max_step:.2e}')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)
    else:
        ax.text(0.5, 0.5, 'No step size history available\n(set track_step_sizes=True)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_ylabel('Step Size')
    ax.set_title('Step Size Evolution Along Curve')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)



def _plot_curvature_evolution(ax, results):
    """Plot the curvature-variation (CurvVar) evolution on the given axis.

    Note: Visualization only displays tracked values; it does not compute metrics.
    """
    # Use tracked CurvVar history if available
    if hasattr(results, 'curvvar_history') and results.curvvar_history and len(results.curvvar_history) > 0:
        x_coords = results.points[:,0]
        curvature_vals = results.curvvar_history

        # Handle different length scenarios
        if len(curvature_vals) == len(x_coords):
            ax.plot(x_coords, curvature_vals, 'm-o', alpha=0.8, linewidth=2, ms=2)
        elif len(curvature_vals) == len(x_coords) - 1:
            ax.plot(x_coords[1:], curvature_vals, 'm-o', alpha=0.8, linewidth=2, ms=2)
        else:
            ax.plot(range(len(curvature_vals)), curvature_vals, 'm-o', alpha=0.8, linewidth=2, ms=2)
            ax.text(0.5, 0.95, f'Warning: curvvar_history length {len(curvature_vals)} does not match points {len(x_coords)}',
                    ha='center', va='top', transform=ax.transAxes, color='red', fontsize=10)

        ax.set_ylabel('CurvVar (tracked)')
    else:
        # No tracked CurvVar available; inform the user rather than compute
        ax.text(0.5, 0.5, 'No CurvVar history available\n(set track_curvvar=True)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
        ax.set_ylabel('CurvVar')

    ax.set_xlabel('x (curve coordinate)')
    ax.set_title('Curvature Variation Along Curve')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)


def plot_trace_result(curve_func, box_min, box_max, p0f, p1f, results,
                     point1_edge, point2_edge, x_padding, y_padding, curve_name):
    """Plot the trace result with curve, radius evolution, and curvature evolution.

    Args:
        curve_func (callable): The curve function to plot
        box_min (np.ndarray): Lower bounds of box
        box_max (np.ndarray): Upper bounds of box
        p0f (np.ndarray): Start point
        p1f (np.ndarray): End point
        results (TraceResult): Results from trace_curve_in_box
        point1_edge (str): Edge for first point
        point2_edge (str): Edge for second point
        x_padding (float): X padding used
        y_padding (float): Y padding used
        curve_name (str): Name of the curve
    """
    import matplotlib.pyplot as plt

    # Create figure with three subplots stacked vertically, sharing x-axis
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Plot each component
    _plot_curve_and_trace(ax1, curve_func, box_min, box_max, p0f, p1f, results, curve_name)
    _plot_radius_evolution(ax2, results)
    _plot_step_size_evolution(ax3, results)
    _plot_curvature_evolution(ax4, results)

    # Set overall title and layout
    fig.suptitle(f'Experiment: {point1_edge}→{point2_edge} | Padding: x={x_padding}, y={y_padding}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()