from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_SHAPE = (50, 96, 96)
EXPECTED_SLICES = np.arange(15, 65, dtype=np.int32)


def _load_patient(path: Path, require_lesion: bool):
    with np.load(path, allow_pickle=False) as data:
        required = {"image", "brain_mask", "patient_id", "slice_indices"}
        if require_lesion:
            required.add("lesion_mask")

        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")

        image = data["image"].astype(np.float32)
        brain_mask = data["brain_mask"].astype(np.uint8)
        lesion_mask = (
            data["lesion_mask"].astype(np.uint8)
            if require_lesion
            else None
        )
        patient_id = str(data["patient_id"].item())
        slice_indices = data["slice_indices"].astype(np.int32)

    if image.shape != EXPECTED_SHAPE or brain_mask.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"{path}: image/brain_mask must have shape {EXPECTED_SHAPE}"
        )

    if lesion_mask is not None and lesion_mask.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"{path}: lesion_mask must have shape {EXPECTED_SHAPE}"
        )

    # 允许 float32 的极小数值误差，例如 1.000000119。
    if (
        not np.isfinite(image).all()
        or image.min() < -1e-6
        or image.max() > 1.0 + 1e-6
    ):
        raise ValueError(f"{path}: image must be finite and in [0, 1]")

    image = np.clip(image, 0.0, 1.0)

    if not np.isin(brain_mask, [0, 1]).all():
        raise ValueError(f"{path}: brain_mask must be binary")

    if lesion_mask is not None and not np.isin(lesion_mask, [0, 1]).all():
        raise ValueError(f"{path}: lesion_mask must be binary")

    if slice_indices.shape != (50,):
        raise ValueError(f"{path}: slice_indices must have length 50")

    if not np.array_equal(slice_indices, EXPECTED_SLICES):
        raise ValueError(f"{path}: slice_indices must be exactly 15..64")

    if patient_id != path.stem:
        raise ValueError(
            f"{path}: filename and patient_id differ ({patient_id})"
        )

    return {
        "image": image,
        "brain_mask": brain_mask,
        "lesion_mask": lesion_mask,
        "patient_id": patient_id,
        "slice_indices": slice_indices,
    }


class BenchmarkTrainSlices(Dataset):
    """
    训练或 validation 使用。
    每次返回一张 [1, 96, 96] slice，数值保持在 [0,1]。
    """

    def __init__(
        self,
        directory,
        horizontal_flip_p=0.0,
        patient_ids=None,
    ):
        self.horizontal_flip_p = float(horizontal_flip_p)
        directory = Path(directory).expanduser()

        paths = sorted(directory.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No NPZ files found in {directory}")

        if patient_ids is not None:
            requested_ids = set(patient_ids)
            available_ids = {path.stem for path in paths}

            missing_ids = requested_ids - available_ids
            if missing_ids:
                raise ValueError(
                    f"{directory}: {len(missing_ids)} requested patient IDs "
                    f"are missing, e.g. {sorted(missing_ids)[:5]}"
                )

            paths = [path for path in paths if path.stem in requested_ids]

            if len(paths) != len(requested_ids):
                raise RuntimeError(
                    f"{directory}: selected {len(paths)} patients, "
                    f"expected {len(requested_ids)}"
                )

        self.patient_ids = [path.stem for path in paths]
        self.items = []

        for path in paths:
            patient = _load_patient(path, require_lesion=False)

            for z in range(50):
                self.items.append(
                    (patient["image"][z], patient["patient_id"], z)
                )

        if not self.items:
            raise RuntimeError("Training dataset is empty")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image, patient_id, z = self.items[index]
        image = torch.from_numpy(image.copy()).unsqueeze(0)

        if torch.rand(()) < self.horizontal_flip_p:
            image = torch.flip(image, dims=[2])

        return {
            "image": image,
            "patient_id": patient_id,
            "slice_index": z,
        }


class BenchmarkPatients(Dataset):
    """Test/predict：一次返回一位患者的全部 50 张切片。"""

    def __init__(self, directory, require_lesion=False):
        self.paths = sorted(Path(directory).expanduser().glob("*.npz"))
        self.require_lesion = require_lesion

        if not self.paths:
            raise FileNotFoundError(f"No NPZ files found in {directory}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        patient = _load_patient(
            self.paths[index],
            require_lesion=self.require_lesion,
        )

        result = {
            "image": torch.from_numpy(patient["image"]),
            "brain_mask": torch.from_numpy(patient["brain_mask"]),
            "patient_id": patient["patient_id"],
            "slice_indices": torch.from_numpy(patient["slice_indices"]),
        }

        if self.require_lesion:
            result["lesion_mask"] = torch.from_numpy(
                patient["lesion_mask"]
            )

        return result