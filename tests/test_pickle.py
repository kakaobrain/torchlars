import copy
import pickle

import pytest
import torch
from torch.optim import SGD

from torchlars import LARS


@pytest.mark.xfail
def test_pickle():
    sgd = SGD([torch.Tensor()], lr=0.1)
    lars = LARS(sgd, trust_coef=0.42)
    lars2 = pickle.loads(pickle.dumps(lars))

    assert isinstance(lars2.optim, SGD)
    assert lars2.optim.param_groups[0]['lr'] == 0.1
    assert lars2.trust_coef == 0.42


@pytest.mark.xfail
def test_copy():
    sgd = SGD([torch.Tensor()], lr=0.1)
    lars = LARS(sgd, trust_coef=0.42)
    lars2 = copy.copy(lars)

    assert isinstance(lars2.optim, SGD)
    assert lars2.optim.param_groups[0]['lr'] == 0.1
    assert lars2.trust_coef == 0.42


@pytest.mark.xfail
def test_deepcopy():
    sgd = SGD([torch.Tensor()], lr=0.1)
    lars = LARS(sgd, trust_coef=0.42)
    lars2 = copy.deepcopy(lars)

    assert isinstance(lars2.optim, SGD)
    assert lars2.optim.param_groups[0]['lr'] == 0.1
    assert lars2.trust_coef == 0.42
