"""Integration tests for the data pipeline.

These wire the real loader, split manifest, transforms and DataLoader together.
They run against the **real GTSRB download when it is present** and are skipped
otherwise, so CI stays fast and free-tier friendly while a developer with the data
gets the stronger check.

The headline assertion is that the split reproduced here is *identical* to the one
recorded in the original COMP9444 run: 21,312 train / 5,328 validation from a
26,640-image training split at seed 43.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from gtsrb.config.schema import ProjectConfig
from gtsrb.data import build_dataloaders, load_raw_split
from gtsrb.data.dataset import LabelledDataset

pytestmark = pytest.mark.integration

#: Size of torchvision's GTSRB ``train`` split. Deliberately not the 39,209 of the
#: official training partition; see docs/PROJECT_AUDIT.md.
EXPECTED_TRAIN_SPLIT_SIZE = 26_640

#: Recorded in the original notebook run.
EXPECTED_SPLIT = {"train": 21_312, "val": 5_328}


def _data_available(root: Path) -> bool:
    return (root / "gtsrb" / "GTSRB" / "Training").exists()


@pytest.fixture
def real_config(tmp_path: Path, project_root: Path) -> ProjectConfig:
    data_root = project_root / "data" / "raw"
    if not _data_available(data_root):
        pytest.skip("GTSRB data not downloaded; run 'make data' to enable this test")
    return ProjectConfig.model_validate(
        {
            "data": {
                "root": str(data_root),
                "image_size": 32,
                "num_workers": 0,
                "split_manifest": str(tmp_path / "split.json"),
            },
            "training": {"batch_size": 8, "device": "cpu", "amp": False},
        }
    )


@pytest.fixture
def raw_train(real_config: ProjectConfig) -> LabelledDataset:
    return load_raw_split(real_config.data.root, "train", download=False)


class TestRealDataset:
    def test_training_split_size_matches_the_recorded_run(self, raw_train: LabelledDataset) -> None:
        assert len(raw_train) == EXPECTED_TRAIN_SPLIT_SIZE
        assert len(raw_train) == 26_640

    def test_test_split_is_the_official_12630(self, real_config: ProjectConfig) -> None:
        assert len(load_raw_split(real_config.data.root, "test", download=False)) == 12_630

    def test_labels_span_all_43_classes(self, raw_train: LabelledDataset) -> None:
        labels = {int(raw_train[index][1]) for index in range(len(raw_train))}
        assert labels == set(range(43))

    def test_images_are_rgb_pil(self, raw_train: LabelledDataset) -> None:
        image, _ = raw_train[0]
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"


class TestRealSplit:
    def test_split_reproduces_the_original_run(self, real_config: ProjectConfig) -> None:
        bundle = build_dataloaders(real_config, download=False)
        assert len(bundle.train_dataset) == EXPECTED_SPLIT["train"]
        assert len(bundle.val_dataset) == EXPECTED_SPLIT["val"]
        assert len(bundle.test_dataset) == 12_630

    def test_train_and_val_are_disjoint(self, real_config: ProjectConfig) -> None:
        bundle = build_dataloaders(real_config, download=False)
        assert set(bundle.train_dataset.indices).isdisjoint(bundle.val_dataset.indices)

    def test_manifest_covers_every_training_sample(self, real_config: ProjectConfig) -> None:
        bundle = build_dataloaders(real_config, download=False)
        covered = set(bundle.manifest.train_indices) | set(bundle.manifest.val_indices)
        assert covered == set(range(EXPECTED_TRAIN_SPLIT_SIZE))

    def test_manifest_is_written_and_reloadable(self, real_config: ProjectConfig) -> None:
        build_dataloaders(real_config, download=False)
        assert real_config.data.split_manifest.exists()
        again = build_dataloaders(real_config, download=False)
        assert again.manifest.num_train == EXPECTED_SPLIT["train"]

    def test_class_balance_is_reported(self, real_config: ProjectConfig) -> None:
        bundle = build_dataloaders(real_config, download=False)
        counts = bundle.manifest.class_counts("val")
        assert set(counts) == set(range(43))
        # GTSRB's rarest class has 150 training images, so its validation share is
        # 30 after a stratified 80/20 split. Assert a loose bound rather than the
        # exact figure so a different torchvision ordering does not break the test.
        assert min(counts.values()) >= 20


class TestRealDataLoader:
    def test_batches_have_the_expected_shape_and_dtype(self, real_config: ProjectConfig) -> None:
        bundle = build_dataloaders(real_config, download=False)
        images, targets = next(iter(bundle.val))
        assert images.shape == (8, 3, 32, 32)
        assert images.dtype == torch.float32
        assert targets.dtype == torch.int64
        assert int(targets.min()) >= 0
        assert int(targets.max()) < 43

    def test_normalised_batches_are_centred(self, real_config: ProjectConfig) -> None:
        """Normalising must centre the data around zero.

        Deliberately measured over a strided sample across the whole validation
        split rather than over the first batch. The validation loader is
        unshuffled and GTSRB is stored class by class, so the first batch holds
        eight images of a single class and is nowhere near representative.

        Without normalisation the mean would sit near the raw pixel mean of ~0.34;
        with normalisation it collapses towards zero.
        """
        bundle = build_dataloaders(real_config, download=False)
        stride = max(1, len(bundle.val_dataset) // 300)
        sample = torch.stack(
            [bundle.val_dataset[i][0] for i in range(0, len(bundle.val_dataset), stride)]
        )
        assert abs(float(sample.mean())) < 0.15
        assert 0.5 < float(sample.std()) < 1.3

    def test_weighted_sampling_raises_rare_class_frequency(
        self, real_config: ProjectConfig
    ) -> None:
        bundle = build_dataloaders(real_config, download=False)
        raw_train = bundle.manifest.class_counts("train")
        assert bundle.class_weights is not None
        effective = bundle.class_weights.effective_counts(
            bundle.train_dataset.labels, len(bundle.train_dataset)
        )
        # The rarest class must gain draws and the most common must lose them.
        rarest = min(raw_train, key=lambda c: raw_train[c])
        commonest = max(raw_train, key=lambda c: raw_train[c])
        assert effective[rarest] > raw_train[rarest]
        assert effective[commonest] < raw_train[commonest]

    def test_evaluation_pipeline_is_deterministic(self, real_config: ProjectConfig) -> None:
        bundle = build_dataloaders(real_config, download=False)
        first = [bundle.val_dataset[i][0] for i in range(3)]
        second = [bundle.val_dataset[i][0] for i in range(3)]
        assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))

    def test_training_pipeline_augments(self, real_config: ProjectConfig) -> None:
        bundle = build_dataloaders(real_config, download=False)
        variants = [bundle.train_dataset[0][0] for _ in range(8)]
        assert any(not torch.equal(variants[0], other) for other in variants[1:])
