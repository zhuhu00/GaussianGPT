import torch

try:
    from pytorch3d.transforms import quaternion_multiply
except ImportError:
    quaternion_multiply = None

from utils.render import GaussianScene, flip_gaussian_scene, rotate_gaussian_scene
from utils.transforms import rotate_points_by_quaternion


def sample_yrotation_quaternion(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Returns a single quaternion representing a random rotation around the y-axis.
    """
    angles = (
        torch.rand((batch_size,), device=device) * 2 * torch.pi
    )  # Random angles in [0, 2*pi), shape (batch_size,)
    return torch.stack(
        [
            torch.cos(angles / 2),
            torch.zeros_like(angles),
            torch.sin(angles / 2),
            torch.zeros_like(angles),
        ],
        dim=-1,
    )  # shape (batch_size, 4)


class RandomYRotation:
    def __init__(self, return_rotation_quaternion: bool = False):
        """
        Data Augmentation that applies a random rotation around the y-axis.
        Will sample a random y-rotation quaternion and apply it to the coordinates and quaternions in the feature dictionary.
        Args:
            return_rotation_quaternion (bool): If True, the sampled rotation quaternion will be added to the feature dictionary under the key 'augmentation_quats'.
            This can be useful for tracking the applied augmentations.
        """
        self.return_rotation_quaternion = return_rotation_quaternion

    def __call__(self, feature_dict: dict) -> dict:
        assert isinstance(
            feature_dict, dict
        ), "Input must be a dictionary containing features. Cannot use tokenized data with augmentations."
        assert (
            feature_dict["coords"].dim() == 2
        ), "Coordinates must be of shape (N, 3) where N is the number of points. The augmentation does not expect a batch dimension."

        num_points = feature_dict["coords"].shape[0]
        device = feature_dict["coords"].device

        # Sample random y-rotation quaternion - we need to apply the same rotation to all points in the scene
        quat = sample_yrotation_quaternion(1, device)  # shape (1, 4)
        quats = quat.repeat(num_points, 1)  # shape (N, 4)

        # Rotate points
        feature_dict["coords"] = rotate_points_by_quaternion(
            quats, feature_dict["coords"]
        )
        if quaternion_multiply is None:
            raise ImportError(
                "RandomYRotation requires pytorch3d. Please install pytorch3d to use this augmentation."
            )
        feature_dict["quats"] = quaternion_multiply(
            quats, feature_dict["quats"]
        )  # careful, not commutative

        if self.return_rotation_quaternion:
            feature_dict["augmentation_quats"] = quat

        return feature_dict


class RandomZAxisDiscreteAugmentation:
    supports_images = True

    def __init__(self, probability: float = 1.0, allow_flip: bool = False):
        if probability < 0.0 or probability > 1.0:
            raise ValueError(
                f"probability must be in [0, 1], got probability={probability}"
            )
        self.probability = float(probability)
        self.allow_flip = bool(allow_flip)

    @staticmethod
    def _z_rotation_matrix(
        steps_90: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        steps_90 %= 4
        if steps_90 == 0:
            return torch.eye(3, device=device, dtype=dtype)
        if steps_90 == 1:
            return torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                device=device,
                dtype=dtype,
            )
        if steps_90 == 2:
            return torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                device=device,
                dtype=dtype,
            )
        return torch.tensor(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _flip_x_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )

    def __call__(self, feature_dict: dict) -> dict:
        assert isinstance(
            feature_dict, dict
        ), "Input must be a dictionary containing features."
        assert (
            "coords" in feature_dict and feature_dict["coords"].dim() == 2
        ), "Coordinates must be of shape (N, 3)."
        assert "quats" in feature_dict, "Expected 'quats' in feature dictionary."

        if self.probability < 1.0 and torch.rand(()) >= self.probability:
            return feature_dict

        device = feature_dict["coords"].device
        steps_90 = int(torch.randint(0, 4, (1,), device=device).item())
        degrees = 90 * steps_90
        do_flip_x = self.allow_flip and bool(
            torch.randint(0, 2, (1,), device=device).item()
        )

        if degrees == 0 and not do_flip_x:
            return feature_dict

        scene = GaussianScene.from_dict(feature_dict)
        if degrees != 0:
            scene = rotate_gaussian_scene(scene, degrees)
        if do_flip_x:
            scene = flip_gaussian_scene(scene, axis="x")
        feature_dict["coords"] = scene.means
        feature_dict["quats"] = scene.quats

        world_transform = self._z_rotation_matrix(
            steps_90, feature_dict["coords"].device, feature_dict["coords"].dtype
        )
        if do_flip_x:
            world_transform = (
                self._flip_x_matrix(
                    feature_dict["coords"].device, feature_dict["coords"].dtype
                )
                @ world_transform
            )

        if "cameras_R" in feature_dict:
            cameras_R = feature_dict["cameras_R"]
            feature_dict["cameras_R"] = torch.matmul(
                world_transform.to(device=cameras_R.device, dtype=cameras_R.dtype),
                cameras_R,
            )

        normals = feature_dict.get("normal")
        if isinstance(normals, torch.Tensor) and normals.shape[-1] == 3:
            feature_dict["normal"] = torch.matmul(
                normals,
                world_transform.to(device=normals.device, dtype=normals.dtype).T,
            )

        return feature_dict
