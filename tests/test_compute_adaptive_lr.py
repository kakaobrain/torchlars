import pytest
import torch

from torchlars._adaptive_lr import compute_adaptive_lr


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_compare_cpu_and_gpu(dtype):
    param_norm = torch.tensor(1., dtype=dtype)
    grad_norm = torch.tensor(1., dtype=dtype)
    adaptive_lr_cpu = torch.tensor(0., dtype=dtype)

    weight_decay = 1.
    eps = 2.
    trust_coef = 1.

    adaptive_lr_cpu = compute_adaptive_lr(
        param_norm,
        grad_norm,
        weight_decay,
        eps,
        trust_coef,
        adaptive_lr_cpu)

    param_norm = torch.tensor(1., dtype=dtype, device='cuda')
    grad_norm = torch.tensor(1., dtype=dtype, device='cuda')
    adaptive_lr_gpu = torch.tensor(0., dtype=dtype, device='cuda')

    weight_decay = 1.
    eps = 2.
    trust_coef = 1.

    adaptive_lr_gpu = compute_adaptive_lr(
        param_norm,
        grad_norm,
        weight_decay,
        eps,
        trust_coef,
        adaptive_lr_gpu)

    assert torch.allclose(adaptive_lr_cpu, adaptive_lr_gpu.cpu())


@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_when_param_norm_is_zero(dtype):
    param_norm = torch.tensor(0., dtype=dtype)
    grad_norm = torch.tensor(1., dtype=dtype)
    adaptive_lr = torch.tensor(0., dtype=dtype)

    weight_decay = 1.
    eps = 1.
    trust_coef = 1.

    adaptive_lr = compute_adaptive_lr(
        param_norm,
        grad_norm,
        weight_decay,
        eps,
        trust_coef,
        adaptive_lr)

    assert adaptive_lr == torch.tensor(1., dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_when_param_norm_is_zero_with_half():
    param_norm = torch.tensor(0., dtype=torch.half, device='cuda')
    grad_norm = torch.tensor(1., dtype=torch.half, device='cuda')
    adaptive_lr = torch.tensor(0., dtype=torch.half, device='cuda')

    weight_decay = 1.
    eps = 1.
    trust_coef = 1.

    adaptive_lr = compute_adaptive_lr(
        param_norm,
        grad_norm,
        weight_decay,
        eps,
        trust_coef,
        adaptive_lr)

    assert adaptive_lr == torch.tensor(1., dtype=torch.half, device='cuda')


@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_when_grad_norm_is_zero(dtype):
    param_norm = torch.tensor(1., dtype=dtype)
    grad_norm = torch.tensor(0., dtype=dtype)
    adaptive_lr = torch.tensor(0., dtype=dtype)

    weight_decay = 1.
    eps = 1.
    trust_coef = 1.

    adaptive_lr = compute_adaptive_lr(
        param_norm,
        grad_norm,
        weight_decay,
        eps,
        trust_coef,
        adaptive_lr)

    assert adaptive_lr == torch.tensor(1., dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_when_grad_norm_is_zero_with_half():
    param_norm = torch.tensor(1., dtype=torch.half, device='cuda')
    grad_norm = torch.tensor(0., dtype=torch.half, device='cuda')
    adaptive_lr = torch.tensor(0., dtype=torch.half, device='cuda')

    weight_decay = 1.
    eps = 1.
    trust_coef = 1.

    adaptive_lr = compute_adaptive_lr(
        param_norm,
        grad_norm,
        weight_decay,
        eps,
        trust_coef,
        adaptive_lr)

    assert adaptive_lr == torch.tensor(1., dtype=torch.half, device='cuda')


@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_specific_case(dtype):
    param_norm = torch.tensor(1.234, dtype=dtype)
    grad_norm = torch.tensor(5.678, dtype=dtype)
    adaptive_lr = torch.tensor(0., dtype=dtype)

    weight_decay = 1e-4
    eps = 1e-8
    trust_coef = 0.001

    adaptive_lr = compute_adaptive_lr(
        param_norm,
        grad_norm,
        weight_decay,
        eps,
        trust_coef,
        adaptive_lr)

    assert torch.allclose(adaptive_lr, torch.tensor(0.000217325, dtype=dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_specific_case_with_half():
    param_norm = torch.tensor(1.234, dtype=torch.half, device='cuda')
    grad_norm = torch.tensor(5.678, dtype=torch.half, device='cuda')
    adaptive_lr = torch.tensor(0., dtype=torch.half, device='cuda')

    weight_decay = 1e-4
    eps = 1e-8
    trust_coef = 0.001

    adaptive_lr = compute_adaptive_lr(
        param_norm,
        grad_norm,
        weight_decay,
        eps,
        trust_coef,
        adaptive_lr)

    assert torch.allclose(adaptive_lr, torch.tensor(0.000217325, dtype=torch.half, device='cuda'))
