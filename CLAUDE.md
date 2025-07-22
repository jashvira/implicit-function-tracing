# Tracing Experiments Project

This project is a comprehensive mathematical curve analysis platform with modular architecture, persistent state management, and multiple visualization options for enhanced boundary handling and intelligent endpoint snapping.

## Project Structure

### Core Modules (`src/`)
- `tracer.py` - Enhanced curve tracing engine with StepState capture and dual-tolerance Newton solver
- `curve_utils.py` - Library of 10 predefined mathematical curves and plotting utilities  
- `state_storage.py` - Complete session persistence system with TraceStateManager

### Visualization System (`visualizers/`)
- `trace_visualizer.py` - Interactive terminal-based step-by-step visualizer
- `simple_visualizer.py` - GUI-based visualizer with matplotlib TkAgg backend

### Interactive Environment
- `tracer.ipynb` - Jupyter notebook for running experiments and saving sessions
- `trace_states/` - Directory for persistent experiment storage (session-based)

## Development Setup

This project uses uv for Python package management with modern pyproject.toml configuration.

### Prerequisites
- Python 3.8+
- uv package manager (installed)

### Dependencies
- numpy - numerical computations
- matplotlib - plotting and visualization
- jupyter - notebook support

### Running Tests
Currently no formal test suite. Testing is done interactively via Jupyter notebook with comprehensive state capture for analysis.

### Code Style
- Follow PEP 8 conventions
- Use type hints where appropriate
- Document complex mathematical algorithms

## Key Components

### TraceConfig
Enhanced configuration class for curve tracing with intelligent parameter tuning:

**Step Control Parameters:**
- `ds_init` (0.1): Initial step size
- `ds_min` (1e-3): Minimum step size before abort
- `ds_max` (0.15): **"Massively controls the crazy newton behaviour, i.e. great for keeping it on leash"**
- `grow` (1.5): Step size growth factor in low-curvature regions

**Convergence Parameters:**
- `arc_eps` (1e-5): Target endpoint proximity tolerance
- `snap_eps` (1e-3): Intelligent endpoint snapping threshold for boundary handling
- `max_iter` (1000): Maximum tracing iterations

**Newton Solver Parameters:**
- `radius_min` (1e-3): Minimum trust region radius
- `shrink` (0.8): **"Gets things moving off the high curvature spots. The bigger, more liberal it is"**
- `max_shrink` (2): Maximum trust region reduction attempts

**State Capture Parameters:**
- `capture_steps` (False): Enable detailed step-by-step state logging for analysis

### Enhanced Newton Solver
- **Dual tolerance system**: Separate `tol` (function) and `dn_tol` (step size) for optimal convergence
- **Performance improvements**: Eliminated duplicate Newton calls with attempt logging
- **Smart boundary handling**: Detects and handles numerical precision issues near box boundaries
- **Trust region optimization**: Enhanced radius management with comprehensive state tracking

### StepState System
- **Complete state capture**: Every Newton iteration tracked with position, predictions, attempts
- **Debugging support**: Detailed failure analysis and convergence diagnostics
- **Persistent storage**: Individual pickle files for each tracing step
- **Interactive analysis**: Step-by-step visualization and inspection

### Intelligent Endpoint Snapping
- **Automatic detection**: When Newton struggles near endpoints, intelligently snaps to target
- **Configurable threshold**: `snap_eps` parameter controls snapping sensitivity
- **Prevents premature termination**: Avoids minimum step size failures due to boundary precision

### Core Functions
- `trace_curve_in_box()` - Main curve tracing function with enhanced boundary handling and state capture
- `newton_tr()` - Enhanced trust-region Newton solver with dual tolerance system
- `adaptive_grad()` - Adaptive gradient computation
- `inward_tangent()` - Box topology helper for direction selection
- `inside()` - Simplified boundary checking with natural operators

### Curve Library (`curve_utils.py`)
Predefined mathematical curves for testing and analysis:
- **Basic waves**: sine, double_sine, damped_sine, sine_cos
- **Polynomials**: cubic with configurable parameters
- **Special curves**: heart, spiral, bumpy (high curvature)
- **Growth functions**: exponential, hyperbola

**Utilities:**
- `calculate_box_bounds()` - Smart bounding box calculation from points and edges
- `plot_trace_result()` - Standardized visualization with curve and trace overlay

### State Management (`state_storage.py`)
- **Session-based persistence**: Complete experiment storage with metadata
- **TraceStateManager**: Handles saving/loading, file I/O, session discovery
- **Step-by-step storage**: Individual pickle files enable detailed analysis
- **Cross-session analysis**: Compare different experiments and parameter sets

### Visualization Options

#### Terminal Visualizer (`trace_visualizer.py`)
- **Interactive navigation**: Arrow keys for frame-by-frame stepping
- **Comprehensive display**: Current position, predictions, Newton attempts, trust regions
- **WSL2 compatible**: Works in headless environments
- **Keyboard controls**: Reset, summary views, step navigation

#### GUI Visualizer (`simple_visualizer.py`)
- **Interactive controls**: Play/pause, frame slider, viewport locking
- **Auto-zoom features**: Intelligent viewport management around active regions
- **Animation support**: Automatic playback of tracing sequences
- **Display detection**: Automatic backend selection based on environment

## Performance Tuning Guide

**For stable tracing on smooth curves:**
- Increase `ds_max` for larger steps
- Increase `shrink` for more liberal trust region behavior
- Enable `capture_steps=True` for detailed analysis

**For handling high curvature regions:**
- Decrease `ds_max` to keep Newton "on leash"
- Tune `shrink` - larger values help escape high curvature spots
- Use curves like `bumpy` or `heart` for testing

**For boundary precision issues:**
- Adjust `snap_eps` - larger values make endpoint snapping more aggressive
- Use `debug=True` to diagnose boundary violations
- Examine saved step states for boundary interaction analysis

**For experiment workflow:**
- Run experiments in Jupyter notebook with state saving
- Use terminal visualizer for detailed step analysis
- Save sessions for reproducible analysis and comparison

## Experiment Workflow

### Typical Experiment Process
1. **Define/select curve** in notebook using `curve_utils` library
2. **Configure tracing** with appropriate `TraceConfig` parameters
3. **Run experiment** with `capture_steps=True` for detailed analysis
4. **Save session** using `TraceStateManager` for persistence
5. **Analyze results** through step states and delta distributions

### Session Management
```python
# Save experiment session
manager = TraceStateManager("experiment_name")
manager.save_session(result, config, curve_info)

# Load and visualize
manager = TraceStateManager("experiment_name")
result, config, metadata = manager.load_session()
```

### Multi-Environment Support
- **Cross-platform**: Jupyter notebook with inline plotting
- **Flexible backends**: Automatic matplotlib backend detection

## Recent Major Improvements

1. **Modular Architecture**: Clean separation of core library (`src/`) and experiments
2. **Persistent State System**: Session-based storage with complete experiment reproducibility  
3. **Comprehensive State Capture**: StepState system enables detailed Newton iteration analysis
4. **Curve Library**: Predefined mathematical functions with smart bounding box calculation
5. **Performance Profiling**: Function evaluation tracking and step-by-step analysis tools
6. **Cross-Session Analysis**: Compare experiments and optimize parameters systematically

## Development Philosophy

- **Separation of concerns**: Clear distinction between computation and storage
- **Reproducibility**: Complete experiment state capture and persistence
- **Extensibility**: Modular architecture supports new curves and analysis tools
- **Robustness**: Focus on numerical stability over mathematical perfection
- **Keep things simple**: Elegant solutions over complex implementations