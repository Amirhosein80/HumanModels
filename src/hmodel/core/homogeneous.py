import torch

__all__ = [
    "get_transformation_matrix",
    "decode_transformation_matrix",
    "inverse_transformation_matrix",
    "skewv",
    "vee",
]


def get_transformation_matrix(
    rotation: torch.Tensor, position: torch.Tensor
) -> torch.Tensor:
    """Build a 4x4 homogeneous transformation matrix.

    Args:
        rotation:  (..., 3, 3) rotation matrix
        position:  (..., 3)    translation vector

    Returns:
        (..., 4, 4) transformation matrix
    """
    # Stack with cat so gradients flow through rotation and position.
    top = torch.cat([rotation, position.unsqueeze(-1)], dim=-1)  # (..., 3, 4)
    bottom_shape = rotation.shape[:-2] + (1, 4)
    bottom = torch.zeros(bottom_shape, dtype=rotation.dtype, device=rotation.device)
    bottom[..., 3] = 1.0
    return torch.cat([top, bottom], dim=-2)  # (..., 4, 4)


def decode_transformation_matrix(transformation: torch.Tensor):
    """Extract rotation and position from a 4x4 homogeneous matrix.

    Args:
        transformation: (..., 4, 4)

    Returns:
        rotation: (..., 3, 3), position: (..., 3)
    """
    rotation = transformation[..., :3, :3]
    position = transformation[..., :3, 3]
    return rotation, position


def inverse_transformation_matrix(transformation: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a rigid-body transformation matrix.

    For T = [R | p; 0 | 1], the inverse is [R^T | -R^T p; 0 | 1].

    Args:
        transformation: (..., 4, 4)

    Returns:
        (..., 4, 4)
    """
    rotation, position = decode_transformation_matrix(transformation)
    rotation_inv = torch.transpose(rotation, -1, -2)
    position_inv = -torch.matmul(rotation_inv, position.unsqueeze(-1)).squeeze(-1)
    return get_transformation_matrix(rotation_inv, position_inv)


def skewv(v: torch.Tensor) -> torch.Tensor:
    """Convert a 3-vector to its 3x3 skew-symmetric (cross-product) matrix.

    skewv(a) @ b  ==  a × b

    Args:
        v: (..., 3)

    Returns:
        (..., 3, 3)
    """
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(x)
    # Stack rows so gradients flow through v.
    return torch.stack(
        [
            torch.stack([O, -z, y], dim=-1),
            torch.stack([z, O, -x], dim=-1),
            torch.stack([-y, x, O], dim=-1),
        ],
        dim=-2,
    )


def vee(mat: torch.Tensor) -> torch.Tensor:
    """Extract the 3-vector from a skew-symmetric matrix (inverse of skewv).

    Args:
        mat: (..., 3, 3) skew-symmetric matrix

    Returns:
        (..., 3)
    """
    return torch.stack([mat[..., 2, 1], mat[..., 0, 2], mat[..., 1, 0]], dim=-1)
