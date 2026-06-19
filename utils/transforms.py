import torch
import torch.nn.functional as F
from pytorch3d.transforms import (
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    standardize_quaternion,
)


def normalize_and_standardize_quaternion(q):
    """Normalize and standardize a quaternion."""
    # q: (*, 4)

    q = q / torch.norm(q, dim=-1, keepdim=True)
    q = standardize_quaternion(q)
    return q


def quaternion_loss(q1, q2):
    """Compute the quaternion distance loss between two quaternions."""
    # q1: (B, 4)
    # q2: (B, 4)
    assert q1.shape == q2.shape, f"Shapes {q1.shape} and {q2.shape} do not match"

    # Normalize the quaternions
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)

    # Compute the quaternion distance
    dot_product = torch.sum(q1 * q2, dim=-1).abs().clamp(0, 1)
    loss = 1 - dot_product

    return loss.mean()


def quaternion_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to 6D rotation representation."""
    # q: (*, 4)
    assert q.shape[-1] == 4, f"Shape {q.shape} is not a quaternion"

    # Normalize the quaternion
    q = normalize_and_standardize_quaternion(q)

    # Convert to rotation matrix
    rot_matrix = quaternion_to_matrix(q)

    # Convert to rotation 6D
    rot_6d = matrix_to_rotation_6d(rot_matrix)

    return rot_6d


def rot6d_to_quaternion(rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to quaternion."""
    # rot_6d: (*, 6)
    assert rot_6d.shape[-1] == 6, f"Shape {rot_6d.shape} is not a rotation 6D"

    # Convert to rotation matrix
    rot_matrix = rotation_6d_to_matrix(rot_6d)

    # Convert to quaternion
    q = matrix_to_quaternion(rot_matrix)

    return q


def rot6d_loss(r1, r2):
    """Compute the L2 loss between two 6D rotation representations. Basically just the MSE between the two rotation matrices."""
    # r1: (*, 6)
    # r2: (*, 6)
    assert r1.shape == r2.shape, f"Shapes {r1.shape} and {r2.shape} do not match"

    # Transform both to full rotation matrices
    r1 = rotation_6d_to_matrix(r1)
    r2 = rotation_6d_to_matrix(r2)

    # Compute the L2 loss between the two rotation matrices
    return F.mse_loss(r1, r2, reduction="mean")


# Covariance matrix
def construct_scaling_rotation_matrix(q, s):
    # q: (B, 4)
    # s: (B, 3)
    assert q.shape[1] == 4, f"Shape {q.shape} is not a quaternion"
    assert s.shape[1] == 3, f"Shape {s.shape} is not a scale"
    assert q.shape[0] == s.shape[0], f"Shapes {q.shape} and {s.shape} do not match"

    # construct the rotation matrix
    rot_matrix = quaternion_to_matrix(q)

    # construct the scale matrix
    scale_matrix = torch.diag_embed(s)

    scaling_rotation = torch.matmul(rot_matrix, scale_matrix)

    return scaling_rotation


def construct_covariance_matrix(q, s):
    # get the scaling rotation matrix
    scaling_rotation = construct_scaling_rotation_matrix(q, s)

    # construct the covariance matrix
    covariance_matrix = torch.matmul(scaling_rotation, scaling_rotation.transpose(1, 2))
    return covariance_matrix


# Taken from 3DGS repository
C0 = 0.28209479177387814


def sh2rgb(sh):
    rgb = sh * C0 + 0.5

    return rgb


def rgb2sh(rgb):
    sh = (rgb - 0.5) / C0

    return sh


def inverse_sigmoid(x):
    # clamp first
    x = torch.clamp(x, 1e-7, 1 - 1e-7)

    return -torch.log(1 / x - 1)


def rotate_points_by_quaternion(
    quats: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    R = quaternion_to_matrix(quats)
    return torch.einsum("bij,bj->bi", R, points)
