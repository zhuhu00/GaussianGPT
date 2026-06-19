import os
import threading
from queue import Queue

import hydra
import lightning
import torch
from tqdm import tqdm

from conf.dataclasses import GaussianFeatures, PCKeys
from data.common import collate_fn
from data.vfront_dataset import VFrontGaussianDataset, get_default_data_split_vfront
from model.gaussian_vqvae import GaussianVQVAE
from utils.gaussian_vqvae_utils import split_batch_dict
from utils.render import (
    GaussianScene,
    center_scene_aabb,
    flip_gaussian_scene,
    render_and_save_trajectory,
    rotate_gaussian_scene,
)

# To enable tensor core usage
torch.set_float32_matmul_precision("high")


def _points_to_scene(points: dict, batch_idx: int = 0) -> GaussianScene:
    batch_mask = points[PCKeys.BATCH] == batch_idx
    return GaussianScene(
        means=points[GaussianFeatures.COORDS][batch_mask],
        opacities=points[GaussianFeatures.OPACITIES][batch_mask],
        scales=points[GaussianFeatures.SCALES][batch_mask],
        quats=points[GaussianFeatures.QUATS][batch_mask],
        sh0=points[GaussianFeatures.SH0][batch_mask],
        sh=points[PCKeys.SH][batch_mask] if PCKeys.SH in points else None,
    )


# Debug config via env vars:
# - GAUSS_DEBUG=1 enables debug mode (default: 0)
# - GAUSS_N_DEBUG_SAMPLES sets number of samples to process (default: 1)
# - GAUSS_N_DEBUG_TOKEN_VIS sets number of reconstructions to render (default: 1)
# - GAUSS_PLOT_STATS=1 writes histogram plots to the working directory (default: 0)
# - GAUSS_MAX_TOKENS skips samples with more than this many tokens (default: disabled)
DEBUG = os.getenv("GAUSS_DEBUG", "0") == "1"
N_DEBUG_SAMPLES = int(os.getenv("GAUSS_N_DEBUG_SAMPLES", "1"))
N_DEBUG_TOKEN_VIS = int(os.getenv("GAUSS_N_DEBUG_TOKEN_VIS", "1"))
PLOT_STATS = os.getenv("GAUSS_PLOT_STATS", "0") == "1"
MAX_TOKENS = (
    int(os.getenv("GAUSS_MAX_TOKENS"))
    if os.getenv("GAUSS_MAX_TOKENS") not in (None, "")
    else None
)
PRINT_STATS_FREQ = 100 if not DEBUG else 1

# used values from configs
# experiment.checkpoint_path must be set
# training.tokenization.output_dir, .sort_by and .generate_augmented_samples (optional) are used
# most values in data.*


def _parse_env_int(name, default=None):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer, got {value}."
        ) from exc


def _resolve_shard(total_items):
    num_shards = _parse_env_int("GAUSS_NUM_SHARDS")
    if num_shards is None:
        num_shards = _parse_env_int("WORLD_SIZE")
    if num_shards is None:
        num_shards = _parse_env_int("SLURM_ARRAY_TASK_COUNT")
    if num_shards is None:
        num_shards = 1

    shard_id = _parse_env_int("GAUSS_SHARD_ID")
    if shard_id is None:
        shard_id = _parse_env_int("RANK")
    if shard_id is None:
        slurm_task_id = _parse_env_int("SLURM_ARRAY_TASK_ID")
        if slurm_task_id is not None:
            slurm_task_min = _parse_env_int("SLURM_ARRAY_TASK_MIN", default=0)
            shard_id = slurm_task_id - slurm_task_min
    if shard_id is None:
        shard_id = 0

    if num_shards < 1:
        raise ValueError(f"Number of shards must be >= 1, got {num_shards}.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Shard id must be in [0, {num_shards - 1}], got {shard_id}.")

    base = total_items // num_shards
    extra = total_items % num_shards
    local_n = base + int(shard_id < extra)
    start = shard_id * base + min(shard_id, extra)
    end = start + local_n
    return shard_id, num_shards, start, end


def _resolve_cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tokenization.")
    local_rank = _parse_env_int("LOCAL_RANK")
    if local_rank is None:
        local_rank = _parse_env_int("SLURM_LOCALID", default=0)
    num_devices = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= num_devices:
        raise ValueError(
            f"LOCAL_RANK={local_rank} is out of range for {num_devices} visible GPUs."
        )
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), local_rank


def _build_tokenization_dataset(cfg, photoshape_sh_degree_override=None):
    dataset_name = str(cfg.data.dataset_name).lower()
    if dataset_name == "vfront":
        split_kwargs = {}
        gaussian_subpath = getattr(cfg.data, "gaussian_subpath", None)
        if gaussian_subpath is not None:
            split_kwargs["gaussian_subpath"] = gaussian_subpath
        train_files, val_files = get_default_data_split_vfront(
            cfg.data.data_path,
            getattr(cfg.data, "train_split", None),
            getattr(cfg.data, "val_split", None),
            verbose=True,
            skip_missing_files=True,
            **split_kwargs,
        )
        files = train_files + val_files
        img_path = getattr(cfg.data, "img_path", None)
        n_images = 1 if img_path is not None else 0
        dataset = VFrontGaussianDataset(
            paths=files,
            split="tokenize",
            transforms_root=getattr(cfg.data, "transforms_path", None),
            load_normals=getattr(cfg.data, "load_normals", False),
            max_points=getattr(cfg.data, "max_points", None),
            n_images=n_images,
            img_path=img_path,
            preload=False,
            sh_degree=getattr(cfg.data, "sh_degree", 0),
            voxel_size=getattr(cfg.data, "voxel_size", 0.025),
            background_color=cfg.data.background_color,
            center_sample=False,  # bool(getattr(cfg.data, "center_sample", True)),
        )
        return files, dataset

    if dataset_name == "ase":
        # Import lazily so VFront-only runs do not require ASE dependencies.
        from data.ase_dataset import ASEGaussianDataset, get_default_data_split_ase

        split_kwargs = {}
        gaussian_subpath = getattr(cfg.data, "gaussian_subpath", None)
        if gaussian_subpath is not None:
            split_kwargs["gaussian_subpath"] = gaussian_subpath
        train_files, val_files = get_default_data_split_ase(
            cfg.data.data_path,
            getattr(cfg.data, "train_split", None),
            getattr(cfg.data, "val_split", None),
            verbose=True,
            skip_missing_files=True,
            **split_kwargs,
        )
        files = train_files + val_files
        dataset = ASEGaussianDataset(
            paths=files,
            split="tokenize",
            transforms_root=getattr(cfg.data, "transforms_path", None),
            transforms_filename=getattr(
                cfg.data, "transforms_filename", "transforms_train.json"
            ),
            load_normals=getattr(cfg.data, "load_normals", False),
            max_points=getattr(cfg.data, "max_points", None),
            n_images=0,
            img_path=getattr(cfg.data, "img_path", None),
            preload=False,
            background_color=cfg.data.background_color,
            center_sample=False,  # bool(getattr(cfg.data, "center_sample", True)),
        )
        return files, dataset

    if dataset_name in {"spp", "spp_v2"}:
        from data.spp_dataset import SPPGaussianDataset, get_default_data_split_spp

        split_kwargs = {}
        gaussian_subpath = getattr(cfg.data, "gaussian_subpath", None)
        if gaussian_subpath is not None:
            split_kwargs["gaussian_subpath"] = gaussian_subpath
        train_files, val_files = get_default_data_split_spp(
            cfg.data.data_path,
            getattr(cfg.data, "train_split", None),
            getattr(cfg.data, "val_split", None),
            verbose=True,
            **split_kwargs,
        )
        files = train_files + val_files
        dataset = SPPGaussianDataset(
            paths=files,
            split="tokenize",
            transforms_root=getattr(cfg.data, "transforms_path", None),
            load_normals=getattr(cfg.data, "load_normals", False),
            max_points=getattr(cfg.data, "max_points", None),
            n_images=1,
            img_path=getattr(cfg.data, "img_path", None),
            preload=False,
            background_color=cfg.data.background_color,
            center_sample=False,
        )
        return files, dataset

    if dataset_name == "photoshape":
        from data.photoshape_dataset import (
            PhotoshapeGaussianDataset,
            get_default_data_split_photoshape,
        )

        split_kwargs = {}
        gaussian_subpath = getattr(cfg.data, "gaussian_subpath", None)
        if gaussian_subpath is not None:
            split_kwargs["gaussian_subpath"] = gaussian_subpath
        train_files, val_files = get_default_data_split_photoshape(
            cfg.data.data_path,
            getattr(cfg.data, "train_split", None),
            getattr(cfg.data, "val_split", None),
            verbose=True,
            **split_kwargs,
        )
        files = train_files + val_files
        photoshape_sh_degree = (
            int(photoshape_sh_degree_override)
            if photoshape_sh_degree_override is not None
            else int(getattr(cfg.data, "sh_degree", 0))
        )
        dataset = PhotoshapeGaussianDataset(
            paths=files,
            split="tokenize",
            transforms_root=getattr(cfg.data, "transforms_path", None),
            transforms_filename=getattr(cfg.data, "transforms_filename", None),
            load_normals=getattr(cfg.data, "load_normals", False),
            max_points=getattr(cfg.data, "max_points", None),
            n_images=0,
            img_path=getattr(cfg.data, "img_path", None),
            preload=False,
            gaussian_subpath=getattr(
                cfg.data,
                "gaussian_subpath",
                "point_cloud/iteration_30000/point_cloud.ply",
            ),
            sh_degree=photoshape_sh_degree,
            background_color=cfg.data.background_color,
            center_sample=False,
        )
        return files, dataset

    raise ValueError(f"Unknown dataset name: {cfg.data.dataset_name}")


@hydra.main(config_path="conf", config_name="vqvae", version_base=None)
def tokenize(cfg):
    assert (
        cfg.experiment.checkpoint_path is not None
    ), "Checkpoint path must be set for tokenized data creation."

    assert (
        cfg.training.tokenization.output_dir is not None
    ), "If create_tokenized_data is set, tokenization.output_dir must be specified."

    print(
        f"INFO: Loading model from {cfg.experiment.checkpoint_path}"
        " - ignoring ModelConfig but updating training parameters"
    )

    print(
        f"INFO: Will sort latents by {cfg.training.tokenization.sort_by} for tokenization."
    )
    generate_augmented_samples = bool(
        getattr(cfg.training.tokenization, "generate_augmented_samples", False)
    )
    if generate_augmented_samples:
        print(
            "INFO: Augmented tokenization enabled "
            "(8 variants per sample: 4 z-rotations x mirrored/non-mirrored)."
        )
    if MAX_TOKENS is not None:
        print(f"INFO: Will skip tokenizations with more than {MAX_TOKENS} tokens.")

    # pylint: disable-next=no-value-for-parameter
    ae = GaussianVQVAE.load_from_checkpoint(
        checkpoint_path=cfg.experiment.checkpoint_path, training_config=cfg.training
    )

    photoshape_sh_degree_override = None
    if str(cfg.data.dataset_name).lower() == "photoshape":
        sh_cfg = ae.autoencoder.features.get("sh")
        expects_sh_input = sh_cfg is not None and bool(sh_cfg.get("in_input", False))
        configured_sh_degree = int(getattr(cfg.data, "sh_degree", 0))
        if expects_sh_input and configured_sh_degree < 1:
            photoshape_sh_degree_override = 1
            print(
                "INFO: Checkpoint expects 'sh' as input feature; "
                "using data.sh_degree=1 for tokenization."
            )

    files, data = _build_tokenization_dataset(
        cfg,
        photoshape_sh_degree_override=photoshape_sh_degree_override,
    )
    shard_id, num_shards, shard_start, shard_end = _resolve_shard(len(files))
    if DEBUG:
        shard_end = min(shard_start + N_DEBUG_SAMPLES, shard_end)
    shard_count = shard_end - shard_start
    shard_label = f"[shard {shard_id + 1}/{num_shards}]"
    print(
        "INFO: Creating tokenized data for GPT training "
        f"{shard_label} with indices [{shard_start}, {shard_end}) "
        f"({shard_count} scenes)."
    )

    os.makedirs(cfg.training.tokenization.output_dir, exist_ok=True)

    # move to selected cuda device and set to eval mode
    device, local_rank = _resolve_cuda_device()
    print(f"INFO: {shard_label} using device={device} (local_rank={local_rank}).")
    ae = ae.to(device).eval()

    # background writer thread to offload torch.save
    def _writer_worker(q: Queue):
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            output_file, tensor = item
            try:
                torch.save(tensor, output_file)
            except Exception as e:  # pylint: disable=broad-except
                print(f"ERROR: Failed to save {output_file}: {e}")
            finally:
                q.task_done()

    save_queue: Queue = Queue(maxsize=32)
    writer_thread = threading.Thread(
        target=_writer_worker, args=(save_queue,), daemon=True
    )
    writer_thread.start()

    point_stats = {
        "min_length": float("inf"),
        "max_length": 0,
        "total_length": 0,
        "count": 0,
    }
    point_lengths = [] if PLOT_STATS else None
    extent_stats = {
        "min_extent": None,
        "max_extent": None,
        "total_extent": torch.zeros(3, dtype=torch.float32),
        "count": 0,
    }
    extent_values = [] if PLOT_STATS else None
    coord_stats = {
        "min_coord": None,
        "max_coord": None,
        "total_min_coord": torch.zeros(3, dtype=torch.float32),
        "total_max_coord": torch.zeros(3, dtype=torch.float32),
        "count": 0,
    }
    coord_min_values = [] if PLOT_STATS else None
    coord_max_values = [] if PLOT_STATS else None
    processed_tokenizations = 0
    considered_tokenizations = 0
    skipped_tokenizations = 0
    skipped_tokenization_names = []
    oom_tokenizations = 0
    oom_tokenization_names = []

    with torch.inference_mode():
        shard_indices = range(shard_start, shard_end)
        data_tqdm = (
            tqdm(shard_indices, total=shard_count) if not DEBUG else shard_indices
        )
        for global_idx in data_tqdm:
            feature_dict = data[global_idx]
            file = files[global_idx]

            feature_dict = collate_fn([feature_dict])

            points, _ = split_batch_dict(feature_dict, device=device)

            points = dict(points)
            base_points_dict = dict(points)
            base_scene = GaussianScene.from_dict(base_points_dict)

            # grab the directory structure above the actual file
            stem = os.path.splitext(os.path.basename(file))[0]
            rel_dir = os.path.relpath(os.path.dirname(file), start=cfg.data.data_path)
            output_dir = os.path.join(cfg.training.tokenization.output_dir, rel_dir)
            rotation_degrees = (0, 90, 180, 270) if generate_augmented_samples else (0,)
            flip_options = (False, True) if generate_augmented_samples else (False,)

            for rotation_deg in rotation_degrees:
                rotated_scene = (
                    base_scene
                    if rotation_deg == 0
                    else rotate_gaussian_scene(base_scene, rotation_deg)
                )

                for flip_x in flip_options:
                    if generate_augmented_samples:
                        suffix = f"_rot{rotation_deg}"
                        if flip_x:
                            suffix += "_flipx"
                    else:
                        suffix = ""
                    variant_name = f"{file}{suffix}"

                    scene = (
                        flip_gaussian_scene(rotated_scene, axis="x")
                        if flip_x
                        else rotated_scene
                    )
                    points_variant = dict(base_points_dict)
                    points_variant["coords"] = scene.means
                    points_variant["quats"] = scene.quats

                    try:
                        tokenized = ae.tokenize(
                            points_variant,
                            sort_latents=cfg.training.tokenization.sort_by,
                        )
                    except (torch.cuda.OutOfMemoryError, MemoryError) as e:
                        oom_tokenizations += 1
                        oom_tokenization_names.append(variant_name)
                        print(
                            "WARNING: OOM during tokenization for "
                            f"{variant_name}: {e}. Skipping.",
                            flush=True,
                        )
                        torch.cuda.empty_cache()
                        continue
                    except RuntimeError as e:
                        oom_message = str(e).lower()
                        if (
                            "out of memory" in oom_message
                            or "cudaerrormemoryallocation" in oom_message
                        ):
                            oom_tokenizations += 1
                            oom_tokenization_names.append(variant_name)
                            print(
                                "WARNING: Runtime OOM during tokenization for "
                                f"{variant_name}: {e}. Skipping.",
                                flush=True,
                            )
                            torch.cuda.empty_cache()
                            continue
                        raise
                    coords = tokenized["coords"].cpu()
                    feature_ids = tokenized["feature_ids"].cpu()

                    num_points = coords.shape[0]
                    if num_points == 0:
                        print(
                            f"WARNING: No points for file {file} "
                            f"(rot={rotation_deg}, flip_x={flip_x}). Skipping."
                        )
                        continue
                    considered_tokenizations += 1
                    if MAX_TOKENS is not None and num_points > MAX_TOKENS:
                        skipped_tokenizations += 1
                        skipped_tokenization_names.append(
                            f"{variant_name} ({num_points} tokens)"
                        )
                        continue

                    point_stats["min_length"] = min(
                        point_stats["min_length"], num_points
                    )
                    point_stats["max_length"] = max(
                        point_stats["max_length"], num_points
                    )
                    point_stats["total_length"] += num_points
                    point_stats["count"] += 1
                    processed_tokenizations += 1
                    if PLOT_STATS:
                        point_lengths.append(int(num_points))

                    coord_min = coords.min(dim=0).values
                    coord_max = coords.max(dim=0).values
                    extent = coord_max - coord_min
                    if extent_stats["min_extent"] is None:
                        extent_stats["min_extent"] = extent.clone()
                        extent_stats["max_extent"] = extent.clone()
                    else:
                        extent_stats["min_extent"] = torch.minimum(
                            extent_stats["min_extent"], extent
                        )
                        extent_stats["max_extent"] = torch.maximum(
                            extent_stats["max_extent"], extent
                        )
                    extent_stats["total_extent"] += extent
                    extent_stats["count"] += 1
                    if PLOT_STATS:
                        extent_values.append(extent.cpu().tolist())
                    if coord_stats["min_coord"] is None:
                        coord_stats["min_coord"] = coord_min.clone()
                        coord_stats["max_coord"] = coord_max.clone()
                    else:
                        coord_stats["min_coord"] = torch.minimum(
                            coord_stats["min_coord"], coord_min
                        )
                        coord_stats["max_coord"] = torch.maximum(
                            coord_stats["max_coord"], coord_max
                        )
                    coord_stats["total_min_coord"] += coord_min
                    coord_stats["total_max_coord"] += coord_max
                    coord_stats["count"] += 1
                    if PLOT_STATS:
                        coord_min_values.append(coord_min.cpu().tolist())
                        coord_max_values.append(coord_max.cpu().tolist())

                    if (
                        PRINT_STATS_FREQ is not None
                        and PRINT_STATS_FREQ > 0
                        and processed_tokenizations % PRINT_STATS_FREQ == 0
                    ):
                        if point_stats["count"] > 0:
                            avg_length = (
                                point_stats["total_length"] / point_stats["count"]
                            )
                            print(
                                f"Point stats after {processed_tokenizations} tokenizations: "
                                f"min={point_stats['min_length']}, "
                                f"max={point_stats['max_length']}, avg={avg_length:.2f}",
                                flush=True,
                            )
                        if extent_stats["count"] > 0:
                            avg_extent = (
                                extent_stats["total_extent"] / extent_stats["count"]
                            )
                            print(
                                "Extent stats after "
                                f"{processed_tokenizations} tokenizations: "
                                f"min={extent_stats['min_extent']}, "
                                f"max={extent_stats['max_extent']}, "
                                f"avg={avg_extent}",
                                flush=True,
                            )
                        if coord_stats["count"] > 0:
                            avg_min_coord = (
                                coord_stats["total_min_coord"] / coord_stats["count"]
                            )
                            avg_max_coord = (
                                coord_stats["total_max_coord"] / coord_stats["count"]
                            )
                            print(
                                "Coord stats after "
                                f"{processed_tokenizations} tokenizations: "
                                f"min={coord_stats['min_coord']}, "
                                f"max={coord_stats['max_coord']}, "
                                f"avg_min={avg_min_coord}, "
                                f"avg_max={avg_max_coord}",
                                flush=True,
                            )
                        if MAX_TOKENS is not None and considered_tokenizations > 0:
                            skipped_pct = (
                                100.0 * skipped_tokenizations / considered_tokenizations
                            )
                            print(
                                f"Skipped {skipped_tokenizations}/{considered_tokenizations} "
                                f"tokenizations due to max tokens "
                                f"({skipped_pct:.2f}%).",
                                flush=True,
                            )

                    if DEBUG:
                        print(
                            f"File: {file}{suffix} -> Coords shape: {coords.shape}, "
                            f"Feature IDs shape: {feature_ids.shape} "
                        )
                        if coords.numel() > 0:
                            n_latents = min(5, coords.shape[0])
                            pairs = list(
                                zip(
                                    coords[:n_latents].tolist(),
                                    feature_ids[:n_latents].tolist(),
                                )
                            )
                            print(
                                f"First {n_latents} latents (coords, feature_ids): {pairs}"
                            )
                            # print min max of coords
                            print(
                                f"Coords range: min={coords.min(dim=0).values.tolist()}, "
                                f"max={coords.max(dim=0).values.tolist()}"
                            )

                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f"{stem}{suffix}.pt")
                    save_queue.put(
                        (output_file, {"coords": coords, "feature_ids": feature_ids})
                    )

                    debug_render_this_variant = DEBUG
                    if debug_render_this_variant:
                        max_num, step_size = coords.shape[0], 1
                        token_lengths_all = list(
                            range(step_size, max_num + step_size, step_size)
                        )

                        # Ensure start and end points are included, and steps are as regular as possible
                        if N_DEBUG_TOKEN_VIS == 1:
                            # just output full reconstruction
                            token_lengths = [token_lengths_all[-1]]
                        elif len(token_lengths_all) > N_DEBUG_TOKEN_VIS:
                            indices = torch.linspace(
                                0, len(token_lengths_all) - 1, N_DEBUG_TOKEN_VIS
                            ).long()
                            token_lengths = [token_lengths_all[idx] for idx in indices]
                            # Ensure start and end are included
                            if token_lengths[0] != token_lengths_all[0]:
                                token_lengths[0] = token_lengths_all[0]
                            if token_lengths[-1] != token_lengths_all[-1]:
                                token_lengths[-1] = token_lengths_all[-1]
                        else:
                            token_lengths = token_lengths_all

                        for j, t in enumerate(token_lengths):
                            coords_t = coords[:t].to(device)
                            feature_ids_t = feature_ids[:t].to(device)

                            autoencoded = ae.decode(coords_t, feature_ids_t)
                            os.makedirs(output_dir, exist_ok=True)
                            render_and_save_trajectory(
                                center_scene_aabb(
                                    _points_to_scene(autoencoded, batch_idx=0)
                                ),
                                os.path.join(
                                    output_dir,
                                    f"{stem}{suffix}_reconstruction_tokens_{t}.gif",
                                ),
                                num_frames=120,
                                fps=24,
                                background_color=cfg.data.background_color,
                            )

                            # render trajectory

    # wait for all queued saves to finish and stop writer
    save_queue.join()
    save_queue.put(None)
    writer_thread.join()

    print("INFO: Tokenization config:", flush=True)
    print(
        f"  feature_tokens={ae.autoencoder.vq.num_tokens}",
        flush=True,
    )
    if PLOT_STATS and point_stats["count"] > 0:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = os.getcwd()
        shard_suffix = (
            "" if num_shards == 1 else f"_shard{shard_id:04d}_of_{num_shards:04d}"
        )
        lengths_path = os.path.join(
            output_dir, f"tokenize_length_hist{shard_suffix}.png"
        )
        extents_path = os.path.join(
            output_dir, f"tokenize_extent_hist{shard_suffix}.png"
        )
        coords_path = os.path.join(
            output_dir, f"tokenize_coord_bounds_hist{shard_suffix}.png"
        )

        fig = plt.figure(figsize=(12, 4), constrained_layout=True)
        if hasattr(fig, "subfigures"):
            subfigs = fig.subfigures(1, 2)
            ax_left = subfigs[0].subplots(1, 1)
            ax_right = subfigs[1].subplots(1, 1)
        else:
            ax_left, ax_right = fig.subplots(1, 2)
        ax_left.hist(point_lengths, bins=50, color="#3B6EA5", alpha=0.9)
        ax_left.set_title("Point Lengths (Linear)")
        ax_left.set_xlabel("Num Points")
        ax_left.set_ylabel("Count")

        ax_right.hist(point_lengths, bins=50, color="#3B6EA5", alpha=0.9, log=True)
        ax_right.set_title("Point Lengths (Log Count)")
        ax_right.set_xlabel("Num Points")
        ax_right.set_ylabel("Count (log)")
        fig.savefig(lengths_path, dpi=150)
        plt.close(fig)

        extent_x = [e[0] for e in extent_values]
        extent_y = [e[1] for e in extent_values]
        extent_z = [e[2] for e in extent_values]

        fig = plt.figure(figsize=(12, 4), constrained_layout=True)
        if hasattr(fig, "subfigures"):
            subfigs = fig.subfigures(1, 3)
            axes = [subfigs[i].subplots(1, 1) for i in range(3)]
        else:
            axes = fig.subplots(1, 3)
        axes[0].hist(extent_x, bins=50, color="#A33B3B", alpha=0.9)
        axes[0].set_title("Extent X")
        axes[0].set_xlabel("Extent")
        axes[0].set_ylabel("Count")
        axes[1].hist(extent_y, bins=50, color="#A33B3B", alpha=0.9)
        axes[1].set_title("Extent Y")
        axes[1].set_xlabel("Extent")
        axes[1].set_ylabel("Count")
        axes[2].hist(extent_z, bins=50, color="#A33B3B", alpha=0.9)
        axes[2].set_title("Extent Z")
        axes[2].set_xlabel("Extent")
        axes[2].set_ylabel("Count")
        fig.savefig(extents_path, dpi=150)
        plt.close(fig)

        coord_min_x = [c[0] for c in coord_min_values]
        coord_min_y = [c[1] for c in coord_min_values]
        coord_min_z = [c[2] for c in coord_min_values]
        coord_max_x = [c[0] for c in coord_max_values]
        coord_max_y = [c[1] for c in coord_max_values]
        coord_max_z = [c[2] for c in coord_max_values]

        fig = plt.figure(figsize=(12, 4), constrained_layout=True)
        if hasattr(fig, "subfigures"):
            subfigs = fig.subfigures(1, 3)
            axes = [subfigs[i].subplots(1, 1) for i in range(3)]
        else:
            axes = fig.subplots(1, 3)
        axes[0].hist(coord_min_x, bins=50, color="#2E8B57", alpha=0.7, label="min")
        axes[0].hist(coord_max_x, bins=50, color="#DAA520", alpha=0.5, label="max")
        axes[0].set_title("Coord Bounds X")
        axes[0].set_xlabel("Coordinate")
        axes[0].set_ylabel("Count")
        axes[0].legend()
        axes[1].hist(coord_min_y, bins=50, color="#2E8B57", alpha=0.7, label="min")
        axes[1].hist(coord_max_y, bins=50, color="#DAA520", alpha=0.5, label="max")
        axes[1].set_title("Coord Bounds Y")
        axes[1].set_xlabel("Coordinate")
        axes[1].set_ylabel("Count")
        axes[1].legend()
        axes[2].hist(coord_min_z, bins=50, color="#2E8B57", alpha=0.7, label="min")
        axes[2].hist(coord_max_z, bins=50, color="#DAA520", alpha=0.5, label="max")
        axes[2].set_title("Coord Bounds Z")
        axes[2].set_xlabel("Coordinate")
        axes[2].set_ylabel("Count")
        axes[2].legend()
        fig.savefig(coords_path, dpi=150)
        plt.close(fig)

        print(
            "INFO: Wrote histogram plots to "
            f"{lengths_path}, {extents_path}, and {coords_path}",
            flush=True,
        )
    if point_stats["count"] == 0:
        print("INFO: No valid point sequences were produced.", flush=True)
    else:
        avg_length = point_stats["total_length"] / point_stats["count"]
        print(
            "INFO: Final point stats: "
            f"min={point_stats['min_length']}, "
            f"max={point_stats['max_length']}, "
            f"avg={avg_length:.2f}, "
            f"samples={point_stats['count']}",
            flush=True,
        )
        if extent_stats["count"] > 0:
            avg_extent = extent_stats["total_extent"] / extent_stats["count"]
            print(
                "INFO: Final extent stats: "
                f"min={extent_stats['min_extent']}, "
                f"max={extent_stats['max_extent']}, "
                f"avg={avg_extent}",
                flush=True,
            )
        if coord_stats["count"] > 0:
            avg_min_coord = coord_stats["total_min_coord"] / coord_stats["count"]
            avg_max_coord = coord_stats["total_max_coord"] / coord_stats["count"]
            print(
                "INFO: Final coord stats: "
                f"min={coord_stats['min_coord']}, "
                f"max={coord_stats['max_coord']}, "
                f"avg_min={avg_min_coord}, "
                f"avg_max={avg_max_coord}",
                flush=True,
            )
    if MAX_TOKENS is not None:
        skipped_pct = (
            100.0 * skipped_tokenizations / considered_tokenizations
            if considered_tokenizations > 0
            else 0.0
        )
        print(
            "INFO: Skipped due to max tokens: "
            f"{skipped_tokenizations}/{considered_tokenizations} "
            f"({skipped_pct:.2f}%).",
            flush=True,
        )
        if skipped_tokenization_names:
            print("INFO: Skipped tokenization names:", flush=True)
            for skipped_name in skipped_tokenization_names:
                print(f"  {skipped_name}", flush=True)
    if oom_tokenizations > 0:
        print(
            f"INFO: Skipped {oom_tokenizations} tokenizations due to OOM errors.",
            flush=True,
        )
        print("INFO: OOM tokenization names:", flush=True)
        for oom_name in oom_tokenization_names:
            print(f"  {oom_name}", flush=True)


if __name__ == "__main__":
    # Seed
    lightning.seed_everything(0)

    tokenize()  # pylint: disable=no-value-for-parameter
