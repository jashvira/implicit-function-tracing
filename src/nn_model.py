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

# Define a simple MLP model using Flax
class MLP(nn.Module):
    num_outputs: int = 2
    num_hidden_units: int = 32
    num_hidden_layers: int = 3

    @nn.compact
    def __call__(self, x):
        # Hidden layers
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_units)(x)
            x = nn.tanh(x)
        # Output layer
        x = nn.Dense(features=self.num_outputs)(x)
        return x

def generate_data(analytical_f: Callable, n_samples: int, key):
    """Generate labeled data from an analytical implicit function."""
    points = jax.random.uniform(key, (n_samples, 2), minval=-1.5, maxval=1.5)
    values = jax.vmap(analytical_f)(points)
    labels = (values > 0).astype(jnp.int32)
    return points, labels

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

def train_model(analytical_f: Callable, n_samples=5000, n_epochs=100, learning_rate=1e-3, seed=42):
    """Train the Flax MLP model on the fly."""
    key = jax.random.PRNGKey(seed)
    mkey, dkey = jax.random.split(key)

    model = MLP()
    train_x, train_y = generate_data(analytical_f, n_samples, dkey)

    params = model.init(mkey, train_x)['params']
    optimizer = optax.adam(learning_rate)

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    for epoch in range(n_epochs):
        state, loss = train_step(state, (train_x, train_y))
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}")

    return state

def get_nn_functions(analytical_f: Callable, train_if_needed: bool = True):
    """
    Returns a pair of functions (nn_f, nn_grad) for a trained neural network.
    """
    model = MLP()

    if train_if_needed:
        print("Training Flax NN model on the fly...")
        state = train_model(analytical_f)
        params = state.params
        print("Training complete.")
    else:
        # Just initialize a dummy model
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))['params']

    @jax.jit
    def nn_f_jax(point: jnp.ndarray, p):
        p_f1, p_f2 = model.apply({'params': p}, point)
        return p_f1 - p_f2

    grad_nn_f = jax.grad(nn_f_jax)

    def nn_f(point: np.ndarray) -> float:
        point_jnp = jnp.asarray(point)
        val = nn_f_jax(point_jnp, params)
        return float(val)

    def nn_grad(point: np.ndarray, counter: list = None) -> np.ndarray:
        if counter is not None:
            counter[0] += 1
        point_jnp = jnp.asarray(point)
        grad_val = grad_nn_f(point_jnp, params)
        return np.asarray(grad_val)

    return nn_f, nn_grad

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