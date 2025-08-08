"""
nn_visualization.py

Neural network visualization utilities for the curve tracing experiments.
Handles plotting of neural network learning quality, training data, and comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Callable, Tuple, Optional

def visualize_nn_learning_quality(
    curve_func: Callable,
    analytical_f: Callable,
    raw_nn_func: Callable,
    nn_grad: Callable,
    box_min: np.ndarray,
    box_max: np.ndarray,
    x_padding: float,
    y_padding: float,
    p0f: np.ndarray,
    p1f: np.ndarray,
    save_path: str = 'nn_learning_comparison.png',
    nn_model_state=None
) -> None:
    """
    Visualise how well the neural network learned the implicit function and its gradient.

    Args:
        curve_func: The curve function y = curve_func(x)
        analytical_f: The analytical implicit function f(x, y)
        raw_nn_func: Callable(point)->[p0, p1] softmax probabilities
        nn_grad: The gradient of the neural network's implicit function
        box_min, box_max: Bounds of the box as 2D arrays
        x_padding, y_padding: Padding for test sampling around the box
        p0f, p1f: Start/end points for overlay
        save_path: If not None, save the figure here
        nn_model_state: Unused; kept for API compatibility
    """
    print("\nVisualising NN learning quality and gradient field...")

    # --- Grid Setup ---
    x_grid_main = np.linspace(box_min[0], box_max[0], 100)
    y_grid_main = np.linspace(box_min[1], box_max[1], 100)
    Xg_main, Yg_main = np.meshgrid(x_grid_main, y_grid_main)
    grid_pts_main = np.column_stack([Xg_main.ravel(), Yg_main.ravel()])

    # --- Plotting Setup ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 12))

    # --- Plot 1: NN Argmax Classification ---
    def _eval_probs(points: np.ndarray) -> np.ndarray:
        probs = [np.asarray(raw_nn_func(p)) for p in points]
        return np.asarray(probs)

    grid_probs = _eval_probs(grid_pts_main)
    nn_classes = np.argmax(grid_probs, axis=1).reshape(Xg_main.shape)

    ax1.contourf(Xg_main, Yg_main, nn_classes, levels=[-0.5, 0.5, 1.5],
                 colors=['lightcoral', 'lightblue'], alpha=0.8)
    ax1.contour(Xg_main, Yg_main, nn_classes, levels=[0.5], colors='black', linewidths=2)

    x_curve = np.linspace(box_min[0], box_max[0], 200)
    y_curve = curve_func(x_curve)
    ax1.plot(x_curve, y_curve, 'k--', linewidth=3, label='True Curve')

    ax1.scatter([p0f[0]], [p0f[1]], c='green', s=100, zorder=10, label='Start')
    ax1.scatter([p1f[0]], [p1f[1]], c='red', s=100, zorder=10, label='End')
    ax1.set_title('NN Argmax Classification')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')

    import matplotlib.patches as mpatches
    outside_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='lightcoral', label='Class 0')
    inside_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Class 1')
    handles, labels = ax1.get_legend_handles_labels()
    handles.extend([outside_patch, inside_patch])
    ax1.legend(handles=handles, loc='best')

    # --- Plot 2: NN Gradient on the Learned Boundary ---
    ax2.set_title('NN Gradient on Learned Boundary')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')

    # Generate the contour data and plot it on the second plot
    contour_set = ax2.contour(Xg_main, Yg_main, nn_classes, levels=[0.5], colors='blue', linewidths=2)

    # Extract points from the contour data
    if contour_set.allsegs and contour_set.allsegs[0]:
        boundary_points = contour_set.allsegs[0][0]
        
        # Select a subset of points to plot gradients for
        step = max(1, len(boundary_points) // 40)  # Show ~40 arrows
        sampled_points = boundary_points[::step]

        # Calculate and plot the gradient at each sampled point
        grads = np.array([nn_grad(p) for p in sampled_points])
        
        # Use autoscaling for quiver plot to represent magnitude
        ax2.quiver(sampled_points[:, 0], sampled_points[:, 1], grads[:, 0], grads[:, 1], 
                   angles='xy', scale_units='xy', scale=None, 
                   color='red', alpha=0.9, width=0.005, headwidth=3)

    # Set plot limits to match the first plot for consistency
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Learned Boundary'),
        Line2D([0], [0], marker='>', color='w', markerfacecolor='r', markersize=10, label='Gradient Vector')
    ]
    ax2.legend(handles=legend_elements, loc='best')

    # --- Final Touches ---
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()

    # --- Lightweight Stats ---
    curve_points = np.column_stack([x_curve, y_curve])
    analytical_on = np.array([analytical_f(p) for p in curve_points])
    nn_probs_on = _eval_probs(curve_points)
    nn_on = nn_probs_on[:, 1] - nn_probs_on[:, 0]

    sign_agree = np.mean((analytical_on >= 0) == (nn_on >= 0))
    print("\nNN Learning Statistics:")
    print(f"   Analytical f range: [{analytical_on.min():.3f}, {analytical_on.max():.3f}]")
    print(f"   NN (p1-p0) range:   [{nn_on.min():.3f}, {nn_on.max():.3f}]")
    print(f"   Boundary agreement on curve: {100.0*sign_agree:.2f}%")


def visualize_training_data(
    viz_points: np.ndarray,
    viz_labels: np.ndarray,
    curve_func: Callable,
    box_min: np.ndarray,
    box_max: np.ndarray,
    x_padding: float,
    y_padding: float,
    p0f: np.ndarray,
    p1f: np.ndarray,
    save_path: str = 'training_data_visualization.png'
) -> None:
    """
    Visualize the training data used for neural network training.

    Args:
        viz_points: Training data points
        viz_labels: Training data labels
        curve_func: The curve function
        box_min: Lower bounds of the box
        box_max: Upper bounds of the box
        x_padding: X padding used
        y_padding: Y padding used
        p0f: Start point
        p1f: End point
        save_path: Path to save the plot
    """
    # Plot the ground truth curve
    fig, ax = plt.subplots(figsize=(8, 6))
    x_curve = np.linspace(box_min[0] - x_padding, box_max[0] + x_padding, 400)
    y_curve = curve_func(x_curve)
    # Convert JAX array to numpy if needed
    if hasattr(y_curve, 'block_until_ready'):
        y_curve = np.asarray(y_curve)
    ax.plot(x_curve, y_curve, 'k--', lw=2, label='Ground Truth Boundary')

    # Separate points by class for better legend
    class_0_mask = viz_labels == 0
    class_1_mask = viz_labels == 1

    class_0_points = viz_points[class_0_mask]
    class_1_points = viz_points[class_1_mask]

    # Scatter plot with explicit class colors and labels
    ax.scatter(class_0_points[:, 0], class_0_points[:, 1], c='purple', alpha=0.6, s=10,
               label=f'Class 0: Outside curve (f > 0) [{np.sum(class_0_mask)} points]')
    ax.scatter(class_1_points[:, 0], class_1_points[:, 1], c='gold', alpha=0.6, s=10,
               label=f'Class 1: Inside curve (f ≤ 0) [{np.sum(class_1_mask)} points]')

    ax.set_title('Generated Training Data for the Neural Network')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(viz_points[:, 0].min(), viz_points[:, 0].max())
    ax.set_ylim(viz_points[:, 1].min(), viz_points[:, 1].max())
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    print(f"Displayed training data visualization with class breakdown")


def debug_nn_functions(
    nn_f: Callable,
    nn_grad: Callable,
    analytical_f: Callable,
    p0f: np.ndarray,
    p1f: np.ndarray
) -> None:
    """
    Debug neural network functions at start and end points.

    Args:
        nn_f: The neural network function
        nn_grad: The neural network gradient function
        analytical_f: The analytical function
        p0f: Start point
        p1f: End point
    """
    print(f"\nDebugging NN functions:")
    print(f"   Start point {p0f}: f={nn_f(p0f):.6f}, grad={nn_grad(p0f)}")
    print(f"   End point {p1f}: f={nn_f(p1f):.6f}, grad={nn_grad(p1f)}")
    print(f"   Analytical f at start: {analytical_f(p0f):.6f}")
    print(f"   Analytical f at end: {analytical_f(p1f):.6f}")


