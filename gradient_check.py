"""
Lightweight test suite for the from-scratch NumPy MLP.

Verifies the analytic gradients produced by functions_back_prop.py against
numerical (finite-difference) gradients on a small toy network, plus a few
shape/sanity checks and a single-batch overfit test.

This imports and exercises your REAL forward/backward prop code — it does
not reimplement the math — so a passing run is a genuine correctness check.

Usage:
    python gradient_check.py          # runs all checks, prints a report
    pytest gradient_check.py -v       # same checks, pytest-style

Place this file at the project root, next to main.py.
"""
import numpy as np
import functions_forward_prop as fp
import functions_back_prop as bp
from classes.network_objects import Neuron, Weight, Bias, Cost_Weight

np.set_printoptions(legacy='1.25')


# --------------------------------------------------------------------------
# Toy network builder
# --------------------------------------------------------------------------

class _ToySettings:
    """Minimal settings stand-in — only the attributes the fp/bp functions
    actually read, kept separate from your real Settings so checks stay
    fast and don't depend on the real dataset."""
    pass


def build_toy_network(seed=0, batch=4, input_width=10, h0=6, h1=5, output_width=3):
    s = _ToySettings()
    s.batch = batch
    s.input_width = input_width
    s.layer0_width = h0
    s.layer1_width = h1
    s.output_width = output_width
    s.learning_rate = 0.05
    s.l2_lambda = 5e-4
    s.beta1, s.beta2, s.epsilon = 0.9, 0.999, 1e-8

    n0, n1, n2 = Neuron(s, h0), Neuron(s, h1), Neuron(s, output_width)
    e0, e1, e2 = Neuron(s, h0), Neuron(s, h1), Neuron(s, output_width)
    w0, w1, w2 = Weight(input_width, h0), Weight(h0, h1), Weight(h1, output_width)
    b0, b1, b2 = Bias(h0), Bias(h1), Bias(output_width)
    cw0 = Cost_Weight(input_width, h0)
    cw1 = Cost_Weight(h0, h1)
    cw2 = Cost_Weight(h1, output_width)

    rng = np.random.RandomState(seed)
    X = rng.randn(batch, input_width).astype(np.float64)
    Y = rng.randint(0, output_width, size=batch)

    return dict(settings=s, X=X, Y=Y,
                n0=n0, n1=n1, n2=n2, e0=e0, e1=e1, e2=e2,
                w0=w0, w1=w1, w2=w2, b0=b0, b1=b1, b2=b2,
                cw0=cw0, cw1=cw1, cw2=cw2)


def forward_and_loss(net):
    s, X, Y = net['settings'], net['X'], net['Y']
    n0, n1, n2 = net['n0'], net['n1'], net['n2']
    w0, w1, w2 = net['w0'], net['w1'], net['w2']
    b0, b1, b2 = net['b0'], net['b1'], net['b2']

    batch = len(X)
    n0.reset(batch); n1.reset(batch); n2.reset(batch)
    n0.layer = fp.hidden_layer_computation(s.layer0_width, s.input_width, X, n0.layer, w0.layer, b0.layer)
    n1.layer = fp.hidden_layer_computation(s.layer1_width, s.layer0_width, n0.layer, n1.layer, w1.layer, b1.layer)
    n2.layer = fp.output_layer_computation(s.output_width, s.layer1_width, n1.layer, n2.layer, w2.layer, b2.layer)

    return fp.loss_function(batch, n2.layer, Y, s, w0, w1, w2)


def analytic_gradients(net):
    """One forward + backward pass; returns gradients without touching parameters."""
    s, X, Y = net['settings'], net['X'], net['Y']
    n0, n1, n2 = net['n0'], net['n1'], net['n2']
    e0, e1, e2 = net['e0'], net['e1'], net['e2']
    w0, w1, w2 = net['w0'], net['w1'], net['w2']
    b0, b1, b2 = net['b0'], net['b1'], net['b2']
    cw0, cw1, cw2 = net['cw0'], net['cw1'], net['cw2']

    batch = len(X)
    n0.reset(batch); n1.reset(batch); n2.reset(batch)
    e0.reset(batch); e1.reset(batch); e2.reset(batch)

    n0.layer = fp.hidden_layer_computation(s.layer0_width, s.input_width, X, n0.layer, w0.layer, b0.layer)
    n1.layer = fp.hidden_layer_computation(s.layer1_width, s.layer0_width, n0.layer, n1.layer, w1.layer, b1.layer)
    n2.layer = fp.output_layer_computation(s.output_width, s.layer1_width, n1.layer, n2.layer, w2.layer, b2.layer)

    e2.layer = bp.error_signal_output(batch, e2.layer, s.output_width, n2.layer, Y)
    e1.layer = bp.error_signal_hidden(e2.layer, s.output_width, e1.layer, n1.layer, s.layer1_width, w2.layer)
    e0.layer = bp.error_signal_hidden(e1.layer, s.layer1_width, e0.layer, n0.layer, s.layer0_width, w1.layer)

    cw2.layer = bp.compute_weight_cost(s, e2.layer, cw2.layer, n1.layer, w2)
    cw1.layer = bp.compute_weight_cost(s, e1.layer, cw1.layer, n0.layer, w1)
    cw0.layer = bp.compute_weight_cost(s, e0.layer, cw0.layer, X, w0)

    cb2 = bp.compute_bias_cost(batch, e2.layer)
    cb1 = bp.compute_bias_cost(batch, e1.layer)
    cb0 = bp.compute_bias_cost(batch, e0.layer)

    return dict(w0=cw0.layer, w1=cw1.layer, w2=cw2.layer, b0=cb0, b1=cb1, b2=cb2)


def numerical_gradient(net, param_name, num_checks=20, epsilon=1e-5, seed=0):
    """Central-difference gradient check on a random subset of a parameter's entries.
    Returns the max relative error found across the sampled entries."""
    param_obj = net[param_name]
    layer = param_obj.layer
    rng = np.random.RandomState(seed)
    idxs = rng.choice(layer.size, size=min(num_checks, layer.size), replace=False)

    analytic = analytic_gradients(net)[param_name]
    max_rel_error = 0.0

    for idx in idxs:
        original = layer.flat[idx]

        layer.flat[idx] = original + epsilon
        loss_plus = forward_and_loss(net)

        layer.flat[idx] = original - epsilon
        loss_minus = forward_and_loss(net)

        layer.flat[idx] = original  # restore

        numeric = (loss_plus - loss_minus) / (2 * epsilon)
        analytic_val = analytic.flat[idx]

        denom = max(abs(numeric), abs(analytic_val), 1e-8)
        max_rel_error = max(max_rel_error, abs(numeric - analytic_val) / denom)

    return max_rel_error


# --------------------------------------------------------------------------
# Tests (pytest-discoverable; also runnable as plain functions)
# --------------------------------------------------------------------------

REL_ERROR_TOLERANCE = 1e-7


def test_gradient_weights():
    net = build_toy_network()
    for name in ['w0', 'w1', 'w2']:
        err = numerical_gradient(net, name)
        assert err < REL_ERROR_TOLERANCE, f"Gradient check failed for {name}: rel error {err:.2e}"


def test_gradient_biases():
    net = build_toy_network()
    for name in ['b0', 'b1', 'b2']:
        err = numerical_gradient(net, name)
        assert err < REL_ERROR_TOLERANCE, f"Gradient check failed for {name}: rel error {err:.2e}"


def test_softmax_output_sums_to_one():
    net = build_toy_network()
    forward_and_loss(net)
    row_sums = net['n2'].layer.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_layer_shapes():
    net = build_toy_network(batch=4, input_width=10, h0=6, h1=5, output_width=3)
    forward_and_loss(net)
    assert net['n0'].layer.shape == (4, 6)
    assert net['n1'].layer.shape == (4, 5)
    assert net['n2'].layer.shape == (4, 3)


def test_overfit_single_batch():
    """Sanity check: with L2 off, loss should collapse when training repeatedly
    on one small batch — a broken update rule or wrong-sign gradient would fail this."""
    net = build_toy_network(seed=1, batch=8, input_width=10, h0=16, h1=16, output_width=3)
    net['settings'].l2_lambda = 0.0  # isolate the fit signal from the regularizer

    initial_loss = forward_and_loss(net)
    t = 0
    for _ in range(200):
        t += 1
        grads = analytic_gradients(net)
        for name in ['w0', 'w1', 'w2', 'b0', 'b1', 'b2']:
            bp.update_parameter(net['settings'], net[name], grads[name], t)
    final_loss = forward_and_loss(net)

    assert final_loss < initial_loss * 0.1, (
        f"Loss did not collapse on a single batch: {initial_loss:.4f} -> {final_loss:.4f}"
    )


# --------------------------------------------------------------------------
# Standalone runner (no pytest required)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_gradient_weights,
        test_gradient_biases,
        test_softmax_output_sums_to_one,
        test_layer_shapes,
        test_overfit_single_batch,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")