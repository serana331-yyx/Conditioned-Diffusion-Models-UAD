import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from src.datamodules.benchmark_dataset import BenchmarkPatients
from src.models.DDPM_2D import gen_noise


EXPECTED_SHAPE = (50, 96, 96)
EXPECTED_SLICES = np.arange(15, 65, dtype=np.int16)
EXPECTED_FOLDS = [1, 2, 3, 4, 5]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_dir: Path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def fold_number_from_checkpoint(path: Path) -> int:
    match = re.search(r"_fold-(\d+)\.ckpt$", path.name)
    if match is None:
        raise ValueError(
            f"{path.name}: expected a checkpoint name ending in _fold-N.ckpt"
        )
    return int(match.group(1))


def sampling_signature(model):
    return {
        "test_timesteps": int(model.test_timesteps),
        "noise_ensemble": bool(model.cfg.get("noise_ensemble", False)),
        "step_ensemble": [
            int(value)
            for value in model.cfg.get("step_ensemble", [250, 500, 750])
        ],
        "noisetype": model.cfg.get("noisetype"),
    }


def load_fold_model(checkpoint: Path, device):
    run_dir = checkpoint.parent.parent
    config_path = run_dir / ".hydra" / "config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Hydra config missing for {checkpoint}: {config_path}"
        )

    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg.model, prefix="")

    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()

    return model, cfg


def reconstruct_one_model(model, images):
    """
    images: float32 [50, 1, 96, 96], values in [0, 1].
    Returns one fold's reconstruction after its internal timestep ensemble.
    """
    features = model(images)

    # MAEDiff requires patch masks; full-image diffusion bypasses its local context.
    if getattr(model, "maediff_enabled", False):
        timesteps = (
            model.cfg.get("step_ensemble", [250, 500, 750])
            if model.cfg.get("noise_ensemble", False)
            else [model.test_timesteps]
        )
        estimates = []
        for timestep in timesteps:
            noise = (
                gen_noise(model.cfg, images.shape).to(images.device)
                if model.cfg.get("noisetype") is not None
                else torch.randn_like(images)
            )
            estimates.append(model._maediff_reconstruct(
                images, features, int(timestep) - 1, noise,
            ))
        return torch.stack(estimates).mean(0).clamp(0.0, 1.0)

    if model.cfg.get("noise_ensemble", False):
        timesteps = model.cfg.get("step_ensemble", [250, 500, 750])
        reconstruction = torch.zeros_like(images)

        for timestep in timesteps:
            noise = (
                gen_noise(model.cfg, images.shape).to(images.device)
                if model.cfg.get("noisetype") is not None
                else None
            )

            _, reco = model.diffusion(
                images,
                cond=features,
                t=int(timestep) - 1,
                noise=noise,
            )
            reconstruction += reco

        reconstruction /= len(timesteps)

    else:
        noise = (
            gen_noise(model.cfg, images.shape).to(images.device)
            if model.cfg.get("noisetype") is not None
            else None
        )

        _, reconstruction = model.diffusion(
            images,
            cond=features,
            t=model.test_timesteps - 1,
            noise=noise,
        )

    return reconstruction.clamp(0.0, 1.0)


def reconstruct_five_fold_ensemble(models, images):
    """
    每个模型先完成自己的 timestep ensemble；
    再对五个 fold reconstruction 做逐像素均值。
    """
    reconstruction_sum = torch.zeros_like(images)

    for model in models:
        reconstruction_sum += reconstruct_one_model(model, images)

    return reconstruction_sum / len(models)


def predict_split(
    models,
    input_dir: Path,
    output_dir: Path,
    expected_count: int,
    dataset_name: str,
    device,
    max_patients=None,
):
    dataset = BenchmarkPatients(input_dir, require_lesion=False)

    if len(dataset) != expected_count:
        raise RuntimeError(
            f"{input_dir}: expected {expected_count} patients, "
            f"found {len(dataset)}"
        )

    patient_total = expected_count
    if max_patients is not None:
        if max_patients < 1:
            raise ValueError("--max_patients must be positive")

        patient_total = min(max_patients, expected_count)
        dataset.paths = dataset.paths[:patient_total]

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    timing_rows = []

    for index, batch in enumerate(loader, start=1):
        patient_id = str(batch["patient_id"][0])
        image = batch["image"][0].numpy().astype(np.float32)
        brain_mask = batch["brain_mask"][0].numpy().astype(np.uint8)
        slice_indices = batch["slice_indices"][0].numpy().astype(np.int16)

        if image.shape != EXPECTED_SHAPE:
            raise RuntimeError(f"{patient_id}: invalid image shape {image.shape}")

        if brain_mask.shape != EXPECTED_SHAPE:
            raise RuntimeError(
                f"{patient_id}: invalid brain_mask shape {brain_mask.shape}"
            )

        if not np.array_equal(slice_indices, EXPECTED_SLICES):
            raise RuntimeError(f"{patient_id}: slice_indices are not 15..64")

        inputs = torch.from_numpy(image).unsqueeze(1).to(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        start_time = time.perf_counter()

        with torch.no_grad():
            reconstruction = reconstruct_five_fold_ensemble(models, inputs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed_seconds = time.perf_counter() - start_time

        peak_gpu_memory_mb = None
        if device.type == "cuda":
            peak_gpu_memory_mb = round(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                2,
            )

        reconstruction = (
            reconstruction.squeeze(1).detach().cpu().numpy().astype(np.float32)
        )
        reconstruction = np.clip(reconstruction, 0.0, 1.0)

        # 经组长确认的 ensemble 输出定义：
        # 先平均五个 fold 的 reconstruction，再计算标准脑区 residual。
        anomaly_map = np.abs(image - reconstruction) * brain_mask
        anomaly_map = np.maximum(anomaly_map, 0.0).astype(np.float32)

        if reconstruction.shape != EXPECTED_SHAPE:
            raise RuntimeError(
                f"{patient_id}: invalid reconstruction shape "
                f"{reconstruction.shape}"
            )

        if anomaly_map.shape != EXPECTED_SHAPE:
            raise RuntimeError(
                f"{patient_id}: invalid anomaly_map shape "
                f"{anomaly_map.shape}"
            )

        if not np.isfinite(reconstruction).all():
            raise RuntimeError(f"{patient_id}: reconstruction has NaN/Inf")

        if not np.isfinite(anomaly_map).all():
            raise RuntimeError(f"{patient_id}: anomaly_map has NaN/Inf")

        if reconstruction.min() < 0.0 or reconstruction.max() > 1.0:
            raise RuntimeError(
                f"{patient_id}: reconstruction outside [0, 1]"
            )

        if anomaly_map.min() < 0.0:
            raise RuntimeError(
                f"{patient_id}: anomaly_map contains negative values"
            )

        np.savez_compressed(
            output_dir / f"{patient_id}.npz",
            patient_id=np.asarray(patient_id),
            slice_indices=slice_indices,
            reconstruction=reconstruction,
            anomaly_map=anomaly_map,
        )

        timing_rows.append(
            {
                "dataset": dataset_name,
                "patient_id": patient_id,
                "wall_time_seconds": round(elapsed_seconds, 6),
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
            }
        )

        print(f"[{index}/{patient_total}] wrote {patient_id}.npz")

    output_files = sorted(output_dir.glob("*.npz"))
    if len(output_files) != patient_total:
        raise RuntimeError(
            f"{output_dir}: expected {patient_total} output NPZ files, "
            f"found {len(output_files)}"
        )

    return timing_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints",
        nargs=5,
        required=True,
        help="Five cDDPM checkpoints, one for each fold.",
    )
    parser.add_argument("--handoff_dir", required=True)
    parser.add_argument("--method_name", default="cDDPM_Spark_5FoldEnsemble")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_patients", type=int, default=None)

    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    handoff_dir = Path(args.handoff_dir).resolve()

    checkpoints = [Path(path).resolve() for path in args.checkpoints]

    if len(set(checkpoints)) != 5:
        raise ValueError("The five checkpoint paths must be distinct")

    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    checkpoints = sorted(checkpoints, key=fold_number_from_checkpoint)
    fold_numbers = [fold_number_from_checkpoint(path) for path in checkpoints]

    if fold_numbers != EXPECTED_FOLDS:
        raise ValueError(
            f"Expected Fold 1..5 checkpoints, got Fold {fold_numbers}"
        )

    if handoff_dir.exists() and any(handoff_dir.iterdir()):
        raise FileExistsError(
            f"{handoff_dir} is not empty. Choose a new empty handoff directory."
        )

    prediction_dir = handoff_dir / "02_test_predictions"
    ixi_output_dir = prediction_dir / "ixi_test"
    brats_output_dir = prediction_dir / "brats2020_test"
    inference_dir = handoff_dir / "03_inference_evidence"

    ixi_output_dir.mkdir(parents=True)
    brats_output_dir.mkdir(parents=True)
    inference_dir.mkdir(parents=True)
    (handoff_dir / "01_training_evidence").mkdir(parents=True)
    (handoff_dir / "integrity").mkdir(parents=True)

    seed_everything(args.seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading five fold models...")

    models = []
    configs = []

    for checkpoint in checkpoints:
        fold_number = fold_number_from_checkpoint(checkpoint)
        print(f"Loading Fold {fold_number}: {checkpoint}")

        model, cfg = load_fold_model(checkpoint, device)
        models.append(model)
        configs.append(cfg)

    signatures = [sampling_signature(model) for model in models]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError(
            "Fold models do not have identical inference parameters: "
            f"{signatures}"
        )

    def get_test_data_paths(cfg):
        """
        Prediction only needs test paths.
        Do not resolve split_file: it contains Hydra runtime interpolation,
        which is not needed by this standalone prediction script.
        """
        return {
            "ixi_test_dir": str(cfg.datamodule.cfg.ixi_test_dir),
            "brats_test_dir": str(cfg.datamodule.cfg.brats_test_dir),
        }

    first_data_cfg = get_test_data_paths(configs[0])

    for cfg in configs[1:]:
        data_cfg = get_test_data_paths(cfg)

        for key in ["ixi_test_dir", "brats_test_dir"]:
            if data_cfg[key] != first_data_cfg[key]:
                raise RuntimeError(
                    f"Fold configs disagree on {key}: "
                    f"{first_data_cfg[key]} vs {data_cfg[key]}"
                )

    run_started = datetime.now(timezone.utc)

    ixi_timing = predict_split(
        models=models,
        input_dir=Path(first_data_cfg["ixi_test_dir"]),
        output_dir=ixi_output_dir,
        expected_count=158,
        dataset_name="ixi_test",
        device=device,
        max_patients=args.max_patients,
    )

    brats_timing = predict_split(
        models=models,
        input_dir=Path(first_data_cfg["brats_test_dir"]),
        output_dir=brats_output_dir,
        expected_count=369,
        dataset_name="brats2020_test",
        device=device,
        max_patients=args.max_patients,
    )

    run_finished = datetime.now(timezone.utc)
    total_prediction_count = len(ixi_timing) + len(brats_timing)
    is_smoke = args.max_patients is not None

    with open(
        inference_dir / "inference_timing.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "patient_id",
                "wall_time_seconds",
                "peak_gpu_memory_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(ixi_timing + brats_timing)

    checkpoint_records = [
        {
            "fold": fold_number_from_checkpoint(path),
            "checkpoint_path": str(path),
            "checkpoint_filename": path.name,
            "checkpoint_sha256": sha256_file(path),
        }
        for path in checkpoints
    ]

    inference_manifest = {
        "method_name": args.method_name,
        "seed": args.seed,
        "checkpoint_type": "five_fold_ensemble",
        "fold_checkpoints": checkpoint_records,
        "selection_rule": (
            "Fixed final step (50000) from each patient-level CV fold; "
            "no test data used for checkpoint selection."
        ),
        "code_repository": str(project_dir),
        "git_commit": git_commit(project_dir),
        "inference_command": " ".join(sys.argv),
        "device": str(device),
        "started_at_utc": run_started.isoformat(),
        "finished_at_utc": run_finished.isoformat(),
        "x_model_normalization": (
            "Input image is supplied in [0,1]; GaussianDiffusion internally "
            "applies x_model = 2 * image - 1."
        ),
        "input_shape": [50, 1, 96, 96],
        "slice_policy": "50 slices, exactly z=15..64",
        "sampling": {
            "within_fold_timestep_ensemble": signatures[0],
            "cross_fold_ensemble": (
                "Mean of five fold reconstructions, one reconstruction "
                "per fold after its own timestep ensemble."
            ),
        },
        "output_definition": {
            "reconstruction_range": [0.0, 1.0],
            "reconstruction": (
                "Mean of five fold reconstructions."
            ),
            "anomaly_map": (
                "abs(input_image - ensemble_reconstruction) * brain_mask"
            ),
        },
        "coverage": {
            "ixi_prediction_count": len(ixi_timing),
            "brats_prediction_count": len(brats_timing),
            "total_prediction_count": total_prediction_count,
            "missing_prediction_count": 0 if not is_smoke else None,
            "run_type": "smoke" if is_smoke else "full_benchmark",
        },
    }

    with open(
        inference_dir / "inference_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(inference_manifest, file, ensure_ascii=False, indent=2)

    submission_manifest = {
        "method_name": args.method_name,
        "seed": args.seed,
        "benchmark_version": "v1.0",
        "ixi_prediction_count": len(ixi_timing),
        "brats_prediction_count": len(brats_timing),
        "total_prediction_count": total_prediction_count,
        "training_data": "IXI train420 only; patient-level 5-fold CV",
        "test_data": "IXI test158 + BraTS2020 test369",
        "contains_test_metrics": False,
        "contains_test_visualizations": False,
        "ensemble": "mean reconstruction over five CV fold models",
        "run_type": "smoke" if is_smoke else "full_benchmark",
    }

    with open(
        handoff_dir / "submission_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(submission_manifest, file, ensure_ascii=False, indent=2)

    readme = f"""# {args.method_name} benchmark handoff

- Seed: {args.seed}
- Training data: IXI train420 only, patient-level 5-fold CV
- Test predictions: IXI test158 and BraTS2020 test369
- Test metrics and test visualizations: not included
- Run type: {"smoke" if is_smoke else "full benchmark"}

## Ensemble definition

For each fold model, reconstructions at timesteps 250, 500 and 750 are
averaged. The final reconstruction is the pixel-wise mean of the five
fold reconstructions. The anomaly map is:

abs(input_image - ensemble_reconstruction) * brain_mask

`02_test_predictions/` contains only the standard prediction NPZ files.
`03_inference_evidence/` records five checkpoints, parameters and timing.
`01_training_evidence/` must be populated from all five Spark and cDDPM runs.
"""
    (handoff_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Five-fold ensemble prediction completed: {handoff_dir}")


if __name__ == "__main__":
    main()
