import json
import os
import time
from typing import NamedTuple, Optional

import numpy as np
import torch
from PIL import Image


def focal2fov(focal_length, image_size):
    return 2 * np.arctan(image_size / (2 * focal_length))


def fov2focal(fov, image_size):
    return image_size / (2 * np.tan(fov / 2))


class SimpleCameraInfo(NamedTuple):
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    cx: float
    cy: float
    image_path: str
    width: int
    height: int


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_num_basis_functions(sh_degree):
    return (sh_degree + 1) ** 2 - 1


def load_pth_sparse_gaussian(path: str, sh_degree: int, voxel_size: float):
    """
    Load sparse gaussian data stored as a .pth file.
    All elemenents are np arrays float32.
    """
    data = safe_torch_load(
        path, map_location="cpu", weights_only=False
    )  # expect a dict

    anchor = data["anchor"]  # [N, 3]

    offset = data["offset"]  # [N, 3]
    offset = 1.5 * voxel_size * np.tanh(offset)

    opacity = data["opacity"]  # [N, 1]
    opacity = np.clip(opacity, -10.0, 10.0)

    scale = data["scale"]  # [N, 3]
    scale = 2 * voxel_size * np_sigmoid(scale)

    rotation = data["rotation"]  # [N, 4]
    rotation = rotation / (
        np.linalg.norm(rotation, axis=-1, keepdims=True) + 1e-8
    )  # normalize

    rgb = data["f_dc"]  # [N, 3]
    rgb = np.clip(rgb, -5.0, 5.0)

    out_dict = {
        "anchor": anchor,
        "offset": offset,
        "rgb": rgb,
        "opacity": opacity,
        "scale": scale,
        "rotation": rotation,
    }

    if sh_degree > 0:
        sh = data["f_rest"]  # [N, 45]
        N = sh.shape[0]
        sh = sh.reshape(N, 3, -1)  # [N, 3, 9]
        sh = sh[
            :, :, : get_num_basis_functions(sh_degree)
        ]  # [N, 3, (sh_degree + 1) ** 2 -1]
        sh = np.transpose(sh, (0, 2, 1))  # [N, (sh_degree + 1) ** 2 -1, 3]
        sh = sh.reshape(N, -1)  # [N, D]
        sh = np.clip(sh, -5.0, 5.0)
        out_dict["sh"] = sh

    return out_dict


def _retry_fs_read(op, path: str, op_name: str, max_retries: int = 10):
    tries = 0
    while True:
        try:
            return op()
        except (PermissionError, OSError, EOFError) as exc:
            # FileNotFoundError can be transient on NFS — retry alongside other OSErrors.
            tries += 1
            if tries > max_retries:
                raise

            print(
                f"{type(exc).__name__} during {op_name} for {path}, "
                f"retrying ({tries}/{max_retries})..."
            )
            time.sleep(min(2 ** (tries - 1), 10.0))


def safe_torch_load(path, map_location=None, weights_only=None):
    # small QoL to avoid random read errors from shared/network filesystems
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    return _retry_fs_read(
        lambda: torch.load(path, **kwargs),
        path,
        "torch.load",
    )


def safe_open_image(image_path):
    # small QoL to avoid random read errors from shared/network filesystems
    def _open():
        image = Image.open(image_path)
        image.load()
        return image

    return _retry_fs_read(_open, image_path, "PIL.Image.open")


def safe_open_json(json_path):
    # small QoL to avoid random read errors from shared/network filesystems
    def _load():
        with open(json_path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)

    return _retry_fs_read(_load, json_path, "json.load")


def safe_read_lines(text_path: str, encoding: Optional[str] = None):
    def _read():
        with open(text_path, "r", encoding=encoding) as text_file:
            return text_file.readlines()

    return _retry_fs_read(_read, text_path, "open(text)")


def safe_cv2_imread(image_path: str, flags):
    import cv2

    def _read():
        image = cv2.imread(image_path, flags)
        if image is None:
            raise OSError(f"cv2.imread returned None for {image_path}.")
        return image

    return _retry_fs_read(_read, image_path, "cv2.imread")


def safe_listdir(path: str):
    return _retry_fs_read(
        lambda: os.listdir(path),
        path,
        "os.listdir",
    )


def safe_exists(path):
    return _retry_fs_read(
        lambda: os.path.exists(path),
        str(path),
        "os.path.exists",
    )


def safe_isdir(path):
    return _retry_fs_read(
        lambda: os.path.isdir(path),
        str(path),
        "os.path.isdir",
    )


def safe_isfile(path):
    return _retry_fs_read(
        lambda: os.path.isfile(path),
        str(path),
        "os.path.isfile",
    )


def readCamerasFromTransforms(
    path,
    transformsfile,
    images_dir,
    extension=".png",
    rotate_y_up=False,
):
    cam_infos = []

    contents = safe_open_json(os.path.join(path, transformsfile))
    if "camera_angle_x" in contents:
        fovx = contents["camera_angle_x"]

    frames = contents["frames"]
    # check if filename already contain postfix
    if frames[0]["file_path"].split(".")[-1] in ["jpg", "jpeg", "JPG", "png"]:
        extension = ""

    for idx, frame in enumerate(frames):

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        if rotate_y_up:
            # Rotate camera frame to y-up convention.
            rot = np.eye(4, dtype=np.float64)
            rot[:3, :3] = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )
            w2c_rotated = np.eye(4, dtype=np.float64)
            w2c_rotated[:3, :3] = R.transpose()  # transpose from glm
            w2c_rotated[:3, 3] = T
            w2c_rotated = w2c_rotated @ rot
            R = w2c_rotated[:3, :3].transpose()  # transpose to glm
            T = w2c_rotated[:3, 3]

        image_path = os.path.join(images_dir, frame["file_path"] + extension)
        if (
            idx == 0
        ):  # read one image to determine size, assuming it is the same for all
            image = safe_open_image(image_path)
            w = image.size[0]
            h = image.size[1]

        if "camera_angle_x" in contents:
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx
        else:
            FovY = focal2fov(contents["fl_y"], image.size[1])
            FovX = focal2fov(contents["fl_x"], image.size[0])
        # prinical point in preprocessed photoshape/vfront scenes seems to be slightly off, better to use centered principal point
        cx = (
            contents["cx"]
            if "cx" in contents
            and not ("photoshape" in path or "vfront" in path or "3dfront" in path)
            else w / 2
        )
        cy = (
            contents["cy"]
            if "cy" in contents
            and not ("photoshape" in path or "vfront" in path or "3dfront" in path)
            else h / 2
        )

        cam_infos.append(
            SimpleCameraInfo(
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                cx=cx,
                cy=cy,
                image_path=image_path,
                width=w,
                height=h,
            )
        )

    return cam_infos
