"""pytorch utilities derived from the VAT code."""
import torch
import torch.nn.functional as F

def multi_sum(input, axes, keepdims=False):
    '''
    Performs `torch.max` over multiple dimensions of `input`
    '''
    axes = sorted(axes)
    sumed = input
    for axis in reversed(axes):
        sumed = sumed.sum(axis, keepdims)
    return sumed


def multi_max(input, axes, keepdims=False):
    '''
    Performs `torch.max` over multiple dimensions of `input`
    '''
    axes = sorted(axes)
    maxed = input
    for axis in reversed(axes):
        maxed, _ = maxed.max(axis, keepdims)
    return maxed

def generate_virtual_adversarial_perturbation(x, logit, forward, vat_epsilon=6.0, vat_xi=1e-6, use_gpu=True):
    """Generate an adversarial perturbation.

    Args:
        x: Model inputs.
        logit: Original model output without perturbation.
        forward: Callable which computs logits given input.
        hps: Model hyperparameters.

    Returns:
        Aversarial perturbation to be applied to x.
    """
    rand_code = torch.randn(x.shape)
    if use_gpu:
        rand_code = rand_code.cuda()

    # todos(jeremy): change 1 to more
    for _ in range(1):
        d = vat_xi * get_normalized_vector(rand_code).clone()

        # use for computing the grad of d
        d.requires_grad = True

        logit_p = logit
        logit_m = forward(x + d)
        dist = torch.mean(kl_divergence_with_logit(logit_p, logit_m))

        grad = torch.autograd.grad(dist, d, only_inputs = True)[0]

        with torch.no_grad():
            d = grad

    return vat_epsilon * get_normalized_vector(d)


def kl_divergence_with_logit(q_logit, p_logit):
    """Compute the per-element KL-divergence of a batch."""
    q = F.softmax(q_logit, 1)
    qlogq = torch.sum(q * logsoftmax(q_logit), 1)
    qlogp = torch.sum(q * logsoftmax(p_logit), 1)
    return qlogq - qlogp


def get_normalized_vector(d):
    """Normalize d by infinity and L2 norms."""
    d /= 1e-12 + multi_max(
        torch.abs(d), list(range(1, len(d.shape))), keepdims=True
    )
    d /= torch.sqrt(
        1e-6
        + multi_sum(
            torch.pow(d, 2.0), list(range(1, len(d.shape))), keepdims=True
        )
    )
    return d

def logsoftmax(x):
    """Compute log-domain softmax of logits."""
    xdev = x - torch.max(x, 1, keepdim=True)[0]
    lsm = xdev - torch.log(torch.sum(torch.exp(xdev), 1, keepdim=True))
    return lsm