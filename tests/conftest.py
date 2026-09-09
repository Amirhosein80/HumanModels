import pytest
import torch

from hmodel.bmodels.smpl import SMPL

MODEL_PATH = "./models"


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture(
    params=[torch.float32, torch.float64, torch.float16, torch.bfloat16],
    ids=["f32", "f64", "f16", "bf16"],
)
def dtype(request):
    return request.param


@pytest.fixture(params=[1, 4], ids=["B1", "B4"])
def batch_size(request):
    return request.param


# Session-scoped: physical_properties is expensive (trimesh + scipy).
# Load each model once and share across all tests in the session.


@pytest.fixture(scope="session")
def smpl_male():
    return SMPL(path=MODEL_PATH, gender="male")


@pytest.fixture(scope="session")
def smpl_female():
    return SMPL(path=MODEL_PATH, gender="female")
