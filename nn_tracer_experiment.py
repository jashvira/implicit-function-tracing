"""
nn_tracer_experiment.py

A script to run the curve tracing experiment using a JAX/Flax neural network model.
This script replicates the logic of the `tracer.ipynb` notebook.
"""

#%%
# CELL 1: IMPORTS AND SETUP
# ----------------------------------------------------------------
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the 'src' directory to the Python path
if 'src' not in sys.path:
    sys.path.append('src')

from src.tracer import TraceConfig, trace_curve_in_box
from src.curve_utils import get_curve_functions, calculate_box_bounds, plot_trace_result
from src.nn_model import get_nn_functions, generate_data

print("All modules loaded successfully.")


#%%
# CELL 2: SETUP CURVE AND TRACING PARAMETERS
# ----------------------------------------------------------------
curve_name = 'double_sine'
curve_func = get_curve_functions()[curve_name]

# Pick points on curve
x_picks = [0.01, 0.9]
points = np.array([[x, curve_func(x)] for x in x_picks])

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
import jax # Needed for PRNGKey

print("\nGenerating and visualizing training data...")

# 1. Define the analytical function that provides the ground truth
analytical_f_for_data = lambda p: p[1] - curve_func(p[0])

# 2. Generate training data points and labels for visualization
# Use a smaller number of samples for a clearer plot
data_key = jax.random.PRNGKey(42)
viz_points, viz_labels = generate_data(analytical_f_for_data, n_samples=2000, key=data_key)

# 3. Plot the ground truth curve
fig, ax = plt.subplots(figsize=(8, 6))
x_curve = np.linspace(box_min[0] - x_padding, box_max[0] + x_padding, 400)
y_curve = curve_func(x_curve)
ax.plot(x_curve, y_curve, 'k--', lw=2, label='Ground Truth Boundary')

# 4. Scatter plot of the generated data
scatter = ax.scatter(viz_points[:, 0], viz_points[:, 1], c=viz_labels, cmap='viridis', alpha=0.6, s=10)
ax.set_title('Generated Training Data for the Neural Network')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(viz_points[:, 0].min(), viz_points[:, 0].max())
ax.set_ylim(viz_points[:, 1].min(), viz_points[:, 1].max())
ax.legend()
plt.show()


#%%
# CELL 3: TRAIN NN AND RUN TRACE
# ----------------------------------------------------------------
print("\nStarting curve trace with NN model...")

config = TraceConfig(
    max_iter=200
)

# 1. Define analytical function for the NN to learn
analytical_f = lambda p: p[1] - curve_func(p[0])

# 2. Get the NN-backed functions (f and gradient)
nn_f, nn_grad = get_nn_functions(analytical_f)

# 3. Run the tracer using the NN functions
results = trace_curve_in_box(
    p0f, p1f, nn_f, box_min, box_max, config,
    gfun_override=nn_grad,
    track_deltas=True,
    track_radii=True,
    track_kappa=True,
    track_step_sizes=True,
    capture_steps=False,
    enable_logging=False
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
                 point1_edge, point2_edge, x_padding, y_padding, curve_name)

print(f"\nPlot shows the traced curve (using NN) from {point1_edge} to {point2_edge} edge.")
plt.show() # Display the plot