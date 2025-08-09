"""
nn_tracer_experiment.py

A script to run the curve tracing experiment using a JAX/Flax neural network model.
This script replicates the logic of the `tracer.ipynb` notebook.
"""

#%%
# CELL 1: IMPORTS AND SETUP
# ----------------------------------------------------------------
import os
# Set environment variable for deterministic operations on Nvidia GPUs
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
# Use a non-interactive backend for matplotlib
os.environ.setdefault('MPLBACKEND', 'Agg')
import jax
jax.config.update("jax_default_matmul_precision", "float32")
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the 'src' directory to the Python path
if 'src' not in sys.path:
    sys.path.append('src')

from src.tracer import TraceConfig, trace_curve_in_box
from src.curve_utils import get_curve_functions, calculate_box_bounds, plot_trace_result
from src.nn_model import get_nn_functions, generate_data
from src.nn_visualization import (
    visualize_nn_learning_quality,
    visualize_training_data,
    debug_nn_functions,
)

print("All modules loaded successfully.")


#%%
# CELL 2: SETUP CURVE AND TRACING PARAMETERS
# ----------------------------------------------------------------
curve_name = 'sine'
curve_func = get_curve_functions()[curve_name]

# Pick points on curve
x_picks = [0.00, 1.00]
points = []
for x in x_picks:
    y = curve_func(x)
    # Convert JAX array to numpy if needed
    if hasattr(y, 'block_until_ready'):
        y = np.asarray(y)
    points.append([x, y])
points = np.array(points)

# Edge configuration
point1_edge = 'left'
point2_edge = 'right'
x_padding = 0.1
y_padding = 2

# Calculate box bounds
box_min, box_max = calculate_box_bounds(points, point1_edge, point2_edge, x_padding, y_padding)

# Final start and end points
p0f = points[0]
p1f = points[1]

print("\nExperiment Configuration:")
print(f"   Curve: {curve_name}")
print(f"   Point 1 on {point1_edge} edge: {p0f}")
print(f"   Point 2 on {point2_edge} edge: {p1f}")
print(f"   Padding: x={x_padding}, y={y_padding}")
print(f"   Box: [{box_min[0]:.3f}, {box_max[0]:.3f}] × [{box_min[1]:.3f}, {box_max[1]:.3f}]")


#%%
# CELL 2.5: VISUALIZE THE GENERATED DATA
# ----------------------------------------------------------------
# %matplotlib widget
import jax # Needed for PRNGKey

print("\nGenerating and visualizing training data...")

# 1. Define the analytical function that provides the ground truth
def analytical_f_for_data(p):
    y_val = curve_func(p[0])
    # Convert JAX array to numpy if needed
    if hasattr(y_val, 'block_until_ready'):
        y_val = np.asarray(y_val)
    return p[1] - y_val

# 2. Generate training data points and labels for visualization
# Increase samples for a much denser, clearer plot
data_key = jax.random.PRNGKey(42)
viz_points, viz_labels = generate_data(analytical_f_for_data, n_samples=20000, key=data_key)

# 3. Visualize training data
visualize_training_data(
    viz_points, viz_labels, curve_func, box_min, box_max,
    x_padding, y_padding, p0f, p1f,
    save_path='docs/training_data.svg'
)

#%%
# CELL 3: TRAIN NN AND RUN TRACE
# ----------------------------------------------------------------
# %matplotlib widget

print("\nStarting curve trace with NN model...")

config = TraceConfig(
    # Rails and termination
    max_iter=1000,
    ds_min=1e-3,
    ds_max=2e-1,
    arc_eps=1e-3,
    # CurvVar damping
    alpha_kappa_var=0.2,
    kappa_var_power=0.3,
    beta_radius=3.5,
    # Newton trust-region policy
    shrink_factor=0.9,
    shrink_iters=5,
    # Hessian bootstrap shaping
    hess_ema_decay=0.9,
    curvvar_eps_factor=0.2,
    curvvar_eps_min=1e-6,
    curvvar_clip=25.0,
    # Step init
    init_step_fraction=0.9,
    # Trust radius lower bound factor
    rho_min_factor=1.2,
    # Damping cap (prevents collapse from spikes)
    curvvar_damping_cap=3.0,
)

# 1. Define analytical function for the NN to learn
def analytical_f(p):
    y_val = curve_func(p[0])
    # Convert JAX array to numpy if needed
    if hasattr(y_val, 'block_until_ready'):
        y_val = np.asarray(y_val)
    return p[1] - y_val

# 2. Get the NN-backed functions (f and gradient)
# Increase n_samples so the NN sees a much denser dataset
nn_f, nn_grad, raw_nn_func = get_nn_functions(analytical_f, n_samples=12000, n_epochs=400)

# Debug: Test the NN functions at the start and end points
debug_nn_functions(nn_f, nn_grad, analytical_f, p0f, p1f)
#%%
# %matplotlib widget
# Visualize how well the NN learned the implicit function
visualize_nn_learning_quality(
    curve_func, analytical_f, raw_nn_func, nn_grad, box_min, box_max,
    x_padding, y_padding, p0f, p1f,
    save_path='docs/nn_learning_comparison.svg'
)
#%%
# 3. Run the tracer using the NN functions
results = trace_curve_in_box(
    p0f, p1f, nn_f, box_min, box_max, config,
    gfun_override=nn_grad,
    f_autodiff=nn_f,
    track_deltas=True,
    track_radii=True,
    track_curvvar=True,
    track_step_sizes=True,
    capture_steps=False,
    enable_logging=True
)

print("\nTrace Results:")
print(f"   Completed: {results.completed}")
print(f"   Reason: {results.reason}")
print(f"   Function evaluations: {results.f_evals}")
print(f"   Traced points: {len(results.points)}")


#%%
# CELL 4: PLOT RESULTS
# ----------------------------------------------------------------
print("\nPlotting results...")
plot_trace_result(curve_func, box_min, box_max, p0f, p1f, results,
                 point1_edge, point2_edge, x_padding, y_padding, curve_name,
                 save_path='docs/trace_result.svg')

