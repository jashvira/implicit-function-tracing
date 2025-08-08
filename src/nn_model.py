"""
nn_model.py ░ v0.2 (Flax version)
----------------------------------------------------------------
JAX/Flax-based neural network model to act as a surrogate for
an implicit function f(x,y)=0.

Key components:
- A simple MLP defined with Flax.
- Data generation from an analytical function.
- A basic training loop using Optax and flax.training.train_state.
- `get_nn_functions` which returns `nn_f` (the implicit function
  f1-f2) and `nn_grad` (its gradient via autograd).
"""
import jax
import jax.numpy as jnp
from flax.training import train_state
import flax.linen as nn
import optax
import numpy as np
from typing import Callable, Any

# Define the ResNet Implicit Softmax model using Flax
class ResBlock(nn.Module):
    width: int
    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(self.width)(x))
        # Zero-init last layer => each block starts as near-identity (Fixup-style).
        h = nn.Dense(self.width, kernel_init=nn.initializers.zeros)(h)
        gamma = self.param('gamma', lambda *_: jnp.array(1.0))  # learnable residual scale
        return x + gamma * h

class ResNetImplicitSoftmax(nn.Module):
    width: int = 256
    blocks: int = 4        # set to your num_hidden_layers

    @nn.compact
    def __call__(self, xy):                # xy: (..., 2)
        h = nn.Dense(self.width)(xy)       # stem
        for _ in range(self.blocks):
            h = ResBlock(self.width)(h)
        s = nn.Dense(1)(nn.relu(h))        # signed field g(x,y)

        logits2 = jnp.concatenate([-s, s], axis=-1)   # (..., 2)
        return nn.softmax(logits2, axis=-1)

def generate_data(analytical_f: Callable, n_samples: int, key):
    """Stratified sampling to densely fill both classes and the boundary.

    Buckets:
    - near-boundary: smallest |f| (with small jitter)
    - inside band: f <= -margin
    - outside band: f >= +margin
    """
    box_min, box_max = -1.5, 1.5
    margin = 0.1
    close_frac, inside_frac, outside_frac = 0.35, 0.35, 0.30

    # PRNG
    k_cand, k_noise_close, k_perm = jax.random.split(key, 3)

    # Candidates
    candidate_count = max(8 * n_samples, n_samples + 256)
    candidates = jax.random.uniform(
        k_cand, (candidate_count, 2), minval=box_min, maxval=box_max
    )
    vals = jax.vmap(analytical_f)(candidates)
    abs_vals = jnp.abs(vals)

    # Quotas
    n_close = int(close_frac * n_samples)
    n_inside = int(inside_frac * n_samples)
    n_out = n_samples - n_close - n_inside

    # Near boundary
    idx_sorted_by_abs = jnp.argsort(abs_vals)
    pts_close = candidates[idx_sorted_by_abs[:n_close]]
    if n_close > 0:
        pts_close = jnp.clip(
            pts_close + 0.02 * jax.random.normal(k_noise_close, pts_close.shape),
            box_min,
            box_max,
        )

    # Inside band (below curve): prefer far-from-boundary negatives
    inside_mask = vals <= -margin
    inside_idx = jnp.nonzero(inside_mask, size=candidate_count)[0]
    pts_inside = candidates[inside_idx[:n_inside]]
    inside_short = n_inside - pts_inside.shape[0]
    if inside_short > 0:
        neg_sorted = jnp.argsort(vals)  # most negative first
        extra_idx = neg_sorted[:inside_short]
        extra_inside = candidates[extra_idx]
        pts_inside = jnp.concatenate([pts_inside, extra_inside], axis=0)

    # Outside band (above curve): prefer far-from-boundary positives
    outside_mask = vals >= margin
    outside_idx = jnp.nonzero(outside_mask, size=candidate_count)[0]
    pts_out = candidates[outside_idx[:n_out]]
    out_short = n_out - pts_out.shape[0]
    if out_short > 0:
        pos_sorted = jnp.argsort(-vals)  # most positive first
        extra_idx = pos_sorted[:out_short]
        extra_out = candidates[extra_idx]
        pts_out = jnp.concatenate([pts_out, extra_out], axis=0)

    # Gather, top-up with more near-boundary if needed
    gathered = [pts_close, pts_inside, pts_out]
    X = jnp.concatenate([g for g in gathered if g.shape[0] > 0], axis=0)
    short = n_samples - X.shape[0]
    if short > 0:
        extra = candidates[idx_sorted_by_abs[n_close : n_close + short]]
        X = jnp.concatenate([X, extra], axis=0)

    # Labels and shuffle
    y = (jax.vmap(analytical_f)(X) <= 0).astype(jnp.int32)
    perm = jax.random.permutation(k_perm, X.shape[0])
    return X[perm], y[perm]

@jax.jit
def train_step(state, batch):
    """A single training step."""
    x, y = batch
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_model(
    analytical_f: Callable,
    n_samples: int = 20000,
    n_epochs: int = 1000,
    learning_rate: float = 3e-4,
    seed: int = 42,
):
    """Train the Flax ResNetImplicitSoftmax model."""
    key = jax.random.PRNGKey(seed)
    mkey, dkey = jax.random.split(key)

    model = ResNetImplicitSoftmax()
    train_x, train_y = generate_data(analytical_f, n_samples, dkey)

    params = model.init(mkey, train_x)['params']
    optimizer = optax.adam(learning_rate)

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    for epoch in range(n_epochs):
        state, loss = train_step(state, (train_x, train_y))
        if epoch % 40 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}")

    return state

def get_nn_functions(
    analytical_f: Callable,
    train_if_needed: bool = True,
    n_samples: int = 20000,
    n_epochs: int = 1000,
    learning_rate: float = 3e-4,
    seed: int = 42,
):
    """
    Returns neural network functions: (nn_f, nn_grad, raw_nn_func).
    - nn_f: f1-f2 implicit function
    - nn_grad: gradient of nn_f
    - raw_nn_func: raw neural network that outputs [f1, f2] probabilities
    """
    model = ResNetImplicitSoftmax()

    if train_if_needed:
        print("Training Flax NN model on the fly...")
        state = train_model(
            analytical_f,
            n_samples=n_samples,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            seed=seed,
        )
        params = state.params
        print("Training complete.")
    else:
        # Just initialize a dummy model
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))['params']

    @jax.jit
    def nn_f_jax(point: jnp.ndarray, p):
        probs = model.apply({'params': p}, point.reshape(1, -1))  # Ensure batch dimension
        f1, f2 = probs[0, 0], probs[0, 1]  # Extract scalars from batch
        return f1 - f2

    @jax.jit
    def raw_nn_jax(point: jnp.ndarray, p):
        """Raw neural network that returns [f1, f2] probabilities."""
        probs = model.apply({'params': p}, point.reshape(1, -1))  # Ensure batch dimension
        return probs[0]  # Return [f1, f2] as 1D array

    grad_nn_f = jax.grad(nn_f_jax)

    def nn_f(point):
        """Unified implicit function.

        - If passed a numpy array, returns a Python float (for NumPy callers).
        - If passed a jax array, returns a JAX scalar (for autodiff/JIT).
        """
        # JAX array path: return JAX scalar for autodiff
        try:
            import jax.numpy as jnp  # type: ignore
            if hasattr(point, 'dtype') and hasattr(point, 'reshape') and 'jax' in type(point).__module__:
                return nn_f_jax(point, params)
        except Exception:
            pass
        # NumPy path: cast to jax array, then return Python float
        point_jnp = jnp.asarray(point)
        val = nn_f_jax(point_jnp, params)
        return float(val)

    def nn_grad(point: np.ndarray, counter: list = None) -> np.ndarray:
        if counter is not None:
            counter[0] += 1
        point_jnp = jnp.asarray(point)
        grad_val = grad_nn_f(point_jnp, params)
        return np.asarray(grad_val)

    def raw_nn_func(point: np.ndarray) -> np.ndarray:
        """Raw neural network function that returns [f1, f2] probabilities."""
        point_jnp = jnp.asarray(point)
        probs = raw_nn_jax(point_jnp, params)
        return np.asarray(probs)

    return nn_f, nn_grad, raw_nn_func

if __name__ == '__main__':
    # 1. Define an analytical function to learn
    def analytical_circle(p):
        return p[0]**2 + p[1]**2 - 0.8

    # 2. Get the NN-backed functions
    nn_f, nn_grad = get_nn_functions(analytical_circle)

    # 3. Test the functions
    test_point = np.array([0.5, 0.5])
    f_val = nn_f(test_point)
    grad_val = nn_grad(test_point)

    print(f"\n--- Verification ---")
    print(f"Analytical f at {test_point}: {analytical_circle(jnp.asarray(test_point))}")
    print(f"NN f at {test_point}: {f_val:.6f}")

    analytical_grad = jax.grad(analytical_circle)(jnp.asarray(test_point))
    print(f"Analytical ∇f at {test_point}: {analytical_grad}")
    print(f"NN ∇f at {test_point}: [{grad_val[0]:.6f}, {grad_val[1]:.6f}]")