"""Runtime workarounds for this machine's CUDA stack.

Kept in one place, and named, so that a future reader can tell a deliberate workaround
from an accident and can delete it once the underlying versions move on.
"""

from __future__ import annotations


def disable_cudnn_rnn_if_unsupported() -> str | None:
    """Fall back to PyTorch's native RNN kernels when cuDNN has none for this GPU.

    On this machine — RTX 5090, compute capability 12.0, torch 2.7.0+cu128 bundling
    cuDNN 9.2.0 — constructing any ``torch.nn.LSTM`` on the GPU raises::

        RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED

    cuDNN 9.2 predates sm_120 and ships no RNN kernels for it. Disabling cuDNN makes
    PyTorch use its own implementation, which is correct and, for the small networks
    here, no slower in any way that matters.

    The switch is global, so it also disables cuDNN for convolutions. That is free today
    because nothing in this project has a convolution — but the camera work planned in
    ``docs/CONTACT_PLAN.md`` would have them, and this call should be made conditional
    (or the CUDA stack upgraded) before that lands.

    Returns a short description of what it did, for the run's provenance record, or
    ``None`` if cuDNN was left alone.
    """
    import torch

    if not torch.cuda.is_available() or not torch.backends.cudnn.is_available():
        return None
    try:
        torch.nn.LSTM(2, 2, 1).cuda()
    except RuntimeError as error:
        if "CUDNN" not in str(error).upper():
            raise
        torch.backends.cudnn.enabled = False
        return (
            f"cudnn disabled: no RNN kernels for sm_"
            f"{''.join(str(v) for v in torch.cuda.get_device_capability(0))} "
            f"in cudnn {torch.backends.cudnn.version()}"
        )
    return None


def zero_initialise_actor_output(runner) -> str:
    """Make the actor emit exactly zero at initialisation.

    Canonical residual policy learning (Silver et al., arXiv:1812.06298 §IV-A) zeroes the
    last layer of the residual network so that::

        "if an initial policy is perfect, then we would like the residual policy to have
         no influence"

    Without it a residual agent opens by perturbing a working controller and has to climb
    back — which is what our first residual run did, starting at 2.52 mm and descending to
    1.37. Silver also warns that a good initial policy paired with an untrained critic
    degrades early, which zeroing does not fix on its own; pairing this with a small
    ``init_noise_std`` is what keeps the opening behaviour close to the base controller.

    Returns a description for the run's provenance record.
    """
    import torch

    actor = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor
    layers = [m for m in actor.mlp.modules() if isinstance(m, torch.nn.Linear)]
    if not layers:
        raise RuntimeError("actor has no Linear layers; cannot zero its output")
    with torch.no_grad():
        layers[-1].weight.zero_()
        if layers[-1].bias is not None:
            layers[-1].bias.zero_()
    return f"actor output layer zero-initialised ({tuple(layers[-1].weight.shape)})"
