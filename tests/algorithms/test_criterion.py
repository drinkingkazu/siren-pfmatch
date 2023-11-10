import torch
import pytest

from pfmatch.algorithms import PoissonMatchLoss
from tests.fixtures import rng, torch_rng, num_pmt

@pytest.fixture
def randn(torch_rng):
    return lambda size, **kwargs: torch.randn(size, generator=torch_rng, **kwargs)

def test_PoissonMatchLoss(num_pmt, randn):
    loss_fn = PoissonMatchLoss()
    qpt_v = randn(size=(2,4))
    qpt_v[1, :3] = 2*qpt_v[0, :3]
    qpt_v[:, -1] = 20_000*qpt_v[:,-1].abs()

    pred = torch.pow(-8*randn(size=(num_pmt,),), 10).requires_grad_(True)
    target = torch.pow(-8*randn(size=(num_pmt,),), 10)
    loss = loss_fn(pred, target)
    assert isinstance(loss, torch.Tensor), 'expected tensor'
    assert loss.shape == (), 'expected scalar'

    loss.backward()