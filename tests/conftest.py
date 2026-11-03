import pytest
import torch

from hmodel.bmodels.smpl import SMPL

MODEL_PATH = "./models"


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


# dtype: float32 is the only parametrized dtype, and all tests compute in it.
#   * Half precision (f16/bf16) has no torch.linalg.qr kernel on CPU or CUDA
#     and cannot meet 1e-5 tolerances, so it is not parametrized here.
#   * float64 is not used at all: assertions compare against scipy / analytic
#     references at float32-appropriate tolerances (1e-4..1e-6).
#   * gradcheck needs eps=1e-4 in float32 — eps=1e-6 is below f32 resolution
#     (1.19e-7) and always yields "Jacobian mismatch".
@pytest.fixture(
    params=[torch.float32],
    ids=["f32"],
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
