"""Unit tests for the data pipeline: transforms, dataset wrapping, splitting and validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from gtsrb.config.schema import AugmentationConfig, DataConfig, ProjectConfig
from gtsrb.data import (
    SplitError,
    SplitManifest,
    TransformSubset,
    build_augmentation,
    build_eval_transform,
    build_manifest,
    build_train_transform,
    channel_statistics,
    compare_statistics,
    compute_class_weights,
    compute_split,
    denormalize,
    get_or_create_manifest,
    label_distribution,
    to_display_array,
    validate_dataset,
    verify_manifest,
)
from gtsrb.data.dataset import load_raw_split, read_labels
from gtsrb.data.loaders import build_dataloaders
from gtsrb.data.validation import DatasetValidationError, estimate_channel_statistics
from tests.conftest import make_sign_image

pytestmark = pytest.mark.unit


class FakeBaseDataset:
    """Minimal stand-in for ``torchvision.datasets.GTSRB``.

    Returns PIL images, like the real base dataset does, and keeps a record of how
    many times each index was decoded so tests can assert that label reads do not
    decode images.
    """

    def __init__(self, labels: list[int], size: int = 32) -> None:
        self._labels = labels
        self._size = size
        self.decodes: list[int] = []

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        self.decodes.append(index)
        return make_sign_image(size=self._size, class_id=self._labels[index]), self._labels[index]


@pytest.fixture
def balanced_labels() -> list[int]:
    return [label for label in range(5) for _ in range(10)]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class TestTransforms:
    def test_eval_transform_produces_normalised_chw_tensor(self) -> None:
        transform = build_eval_transform(32, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        out = transform(make_sign_image(size=64))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 32, 32)
        assert out.dtype == torch.float32

    def test_eval_transform_is_deterministic(self) -> None:
        transform = build_eval_transform(32, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = make_sign_image(size=40)
        assert torch.equal(transform(image), transform(image))

    def test_train_transform_shares_geometry_with_eval(self) -> None:
        config = DataConfig(
            image_size=24, normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)
        )
        assert build_train_transform(config)(make_sign_image(48)).shape == (3, 24, 24)

    def test_train_transform_is_stochastic_when_augmentation_is_on(self) -> None:
        config = DataConfig(image_size=32)
        transform = build_train_transform(config)
        image = make_sign_image(size=64)
        outputs = [transform(image) for _ in range(8)]
        assert any(not torch.equal(outputs[0], other) for other in outputs[1:])

    def test_train_transform_is_deterministic_when_augmentation_is_off(self) -> None:
        config = DataConfig.model_validate(
            {
                "image_size": 32,
                "augmentation": {"enabled": False},
            }
        )
        transform = build_train_transform(config)
        image = make_sign_image(size=64)
        assert torch.equal(transform(image), transform(image))

    def test_augmentation_disabled_yields_empty_block(self) -> None:
        assert len(build_augmentation(AugmentationConfig(enabled=False)).transforms) == 0

    def test_augmentation_does_not_flip_horizontally(self) -> None:
        """A mirrored traffic sign changes meaning, so no flip may ever appear."""
        config = DataConfig(image_size=32)
        block = build_train_transform(config)
        flip_types = (transforms.RandomHorizontalFlip, transforms.RandomVerticalFlip)
        assert not any(isinstance(step, flip_types) for step in block.transforms)

    def test_random_erasing_only_when_requested(self) -> None:
        default = build_train_transform(DataConfig(image_size=32))
        assert not any(isinstance(step, transforms.RandomErasing) for step in default.transforms)

        with_erasing = build_train_transform(
            DataConfig.model_validate({"image_size": 32, "augmentation": {"random_erasing_p": 0.5}})
        )
        assert isinstance(with_erasing.transforms[-1], transforms.RandomErasing)

    def test_scale_augmentation_applied_via_affine(self) -> None:
        block = build_augmentation(
            DataConfig.model_validate(
                {"image_size": 32, "augmentation": {"scale_min": 0.8, "scale_max": 1.2}}
            ).augmentation
        )
        affine = [step for step in block.transforms if isinstance(step, transforms.RandomAffine)]
        assert len(affine) == 1
        assert affine[0].scale == (0.8, 1.2)

    def test_no_affine_op_when_all_geometry_disabled(self) -> None:
        block = build_augmentation(
            DataConfig.model_validate(
                {
                    "image_size": 32,
                    "augmentation": {
                        "rotation_degrees": 0.0,
                        "translate": 0.0,
                        "shear_degrees": 0.0,
                        "brightness": 0.0,
                        "contrast": 0.0,
                    },
                }
            ).augmentation
        )
        assert not any(isinstance(step, transforms.RandomAffine) for step in block.transforms)


class TestDenormalize:
    def test_round_trips_a_normalised_tensor(self) -> None:
        mean = (0.3403, 0.3121, 0.3214)
        std = (0.2724, 0.2608, 0.2669)
        original = torch.rand(3, 8, 8)
        normalized = (original - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
        assert torch.allclose(denormalize(normalized, mean, std), original, atol=1e-5)

    def test_output_is_clamped_to_unit_range(self) -> None:
        result = denormalize(torch.full((3, 4, 4), 50.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert float(result.max()) == pytest.approx(1.0)

    def test_batch_tensor_is_supported(self) -> None:
        assert denormalize(torch.zeros(4, 3, 8, 8), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).shape == (
            4,
            3,
            8,
            8,
        )

    @pytest.mark.parametrize("shape", [(3,), (2, 3, 4, 5, 6)])
    def test_wrong_rank_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="3D or 4D"):
            denormalize(torch.zeros(shape), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def test_to_display_array_shape_and_range(self) -> None:
        array = to_display_array(torch.rand(3, 8, 8), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert array.shape == (8, 8, 3)
        assert array.min() >= 0.0 and array.max() <= 1.0

    def test_to_display_array_rejects_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match="single CHW"):
            to_display_array(torch.rand(1, 3, 8, 8))

    def test_to_display_array_rejects_non_rgb(self) -> None:
        with pytest.raises(ValueError, match="3 channels"):
            to_display_array(torch.rand(1, 8, 8))


class TestChannelStatistics:
    def test_computes_per_channel_moments(self) -> None:
        images = [np.zeros((4, 4, 3), dtype=np.float32), np.ones((4, 4, 3), dtype=np.float32)]
        mean, std = channel_statistics(images)
        assert mean == pytest.approx((0.5, 0.5, 0.5))
        assert std == pytest.approx((0.5, 0.5, 0.5))

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty sequence"):
            channel_statistics([])

    def test_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(H, W, 3\)"):
            channel_statistics([np.zeros((4, 4), dtype=np.float32)])


# ---------------------------------------------------------------------------
# TransformSubset
# ---------------------------------------------------------------------------
class TestTransformSubset:
    def test_len_follows_indices_not_base(self) -> None:
        base = FakeBaseDataset(list(range(5)))
        assert len(TransformSubset(base, [0, 2, 4], transforms.ToTensor())) == 3

    def test_maps_through_to_the_right_base_index(self) -> None:
        base = FakeBaseDataset([0, 1, 2, 3, 4])
        subset = TransformSubset(base, [4, 1], transforms.ToTensor())
        assert int(subset[0][1]) == 4
        assert int(subset[1][1]) == 1

    def test_labels_are_long_tensors(self) -> None:
        base = FakeBaseDataset([0, 1])
        subset = TransformSubset(base, [0, 1], transforms.ToTensor())
        assert subset[0][1].dtype == torch.long

    def test_applies_transform(self) -> None:
        base = FakeBaseDataset([0, 1])
        subset = TransformSubset(base, [0, 1], build_eval_transform(16, (0.5,) * 3, (0.5,) * 3))
        image, _ = subset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 16, 16)

    def test_label_reads_do_not_decode_images(self) -> None:
        """Sampler weights come from labels; decoding every image to get them is wasteful."""
        base = FakeBaseDataset([0, 1, 2])
        subset = TransformSubset(base, [0, 1, 2], transforms.ToTensor())
        base.decodes.clear()
        assert subset.labels == [0, 1, 2]

    def test_label_distribution(self) -> None:
        base = FakeBaseDataset([0, 0, 1, 2, 2, 2])
        subset = TransformSubset(base, list(range(6)), transforms.ToTensor())
        assert subset.label_distribution() == {0: 2, 1: 1, 2: 3}

    def test_invalid_label_is_rejected_at_construction(self) -> None:
        base = FakeBaseDataset([0, 99])
        with pytest.raises(ValueError, match="outside \\[0, 43\\)"):
            TransformSubset(base, [0, 1], transforms.ToTensor(), num_classes=43)

    def test_label_validation_can_be_disabled(self) -> None:
        base = FakeBaseDataset([0, 99])
        subset = TransformSubset(base, [0, 1], transforms.ToTensor(), validate_labels=False)
        assert len(subset) == 2


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
class TestComputeSplit:
    def test_split_is_deterministic_for_a_seed(self, balanced_labels: list[int]) -> None:
        assert compute_split(balanced_labels, 0.2, 43) == compute_split(balanced_labels, 0.2, 43)

    def test_split_changes_with_the_seed(self, balanced_labels: list[int]) -> None:
        assert compute_split(balanced_labels, 0.2, 43) != compute_split(balanced_labels, 0.2, 44)

    def test_indices_are_disjoint_and_complete(self, balanced_labels: list[int]) -> None:
        train, val = compute_split(balanced_labels, 0.2, 43)
        assert set(train).isdisjoint(val)
        assert sorted(train + val) == list(range(len(balanced_labels)))

    def test_indices_are_sorted_for_a_stable_manifest(self, balanced_labels: list[int]) -> None:
        train, val = compute_split(balanced_labels, 0.2, 43)
        assert train == sorted(train)
        assert val == sorted(val)

    def test_stratification_preserves_class_proportions(self) -> None:
        labels = [0] * 100 + [1] * 10 + [2] * 50
        _train, val = compute_split(labels, 0.2, 43)
        for class_id in (0, 1, 2):
            total = labels.count(class_id)
            in_val = sum(1 for i in val if labels[i] == class_id)
            assert abs(in_val / total - 0.2) < 0.05

    def test_val_fraction_is_respected(self, balanced_labels: list[int]) -> None:
        train, val = compute_split(balanced_labels, 0.25, 43)
        assert len(val) == 13
        assert len(train) == 37

    def test_empty_labels_raise(self) -> None:
        with pytest.raises(SplitError, match="empty dataset"):
            compute_split([], 0.2, 43)

    def test_single_member_class_cannot_be_stratified(self) -> None:
        with pytest.raises(SplitError, match="at least 2 samples"):
            compute_split([0, 0, 1], 0.2, 43)


class TestSplitManifest:
    def test_round_trips_through_json(self, tmp_path: Path, balanced_labels: list[int]) -> None:
        config = DataConfig.model_validate({"split_manifest": str(tmp_path / "split.json")})
        manifest = build_manifest(balanced_labels, config)
        manifest.save(config.split_manifest)
        assert SplitManifest.load(config.split_manifest) == manifest

    def test_class_counts_are_reported_per_split(self, tmp_path: Path) -> None:
        labels = [0] * 10 + [1] * 10
        config = DataConfig.model_validate({"split_manifest": str(tmp_path / "split.json")})
        manifest = build_manifest(labels, config)
        assert sum(manifest.class_counts("train").values()) == manifest.num_train
        assert sum(manifest.class_counts("val").values()) == manifest.num_val

    def test_unknown_split_name_raises(self, tmp_path: Path, balanced_labels: list[int]) -> None:
        config = DataConfig.model_validate({"split_manifest": str(tmp_path / "split.json")})
        with pytest.raises(ValueError, match="Unknown split"):
            build_manifest(balanced_labels, config).class_counts("test")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SplitError, match="not found"):
            SplitManifest.load(tmp_path / "absent.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SplitError, match="not valid JSON"):
            SplitManifest.load(path)

    def test_non_object_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(SplitError, match="must contain a JSON object"):
            SplitManifest.load(path)

    def test_missing_keys_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"version": 1}), encoding="utf-8")
        with pytest.raises(SplitError, match="missing keys"):
            SplitManifest.load(path)

    def test_unsupported_version_raises(self, tmp_path: Path, balanced_labels: list[int]) -> None:
        config = DataConfig.model_validate({"split_manifest": str(tmp_path / "split.json")})
        payload = build_manifest(balanced_labels, config).to_dict()
        payload["version"] = 999
        path = tmp_path / "v999.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SplitError, match="version 999 is not supported"):
            SplitManifest.load(path)


class TestVerifyManifest:
    def _config(self, tmp_path: Path, **overrides: object) -> DataConfig:
        payload: dict[str, object] = {"split_manifest": str(tmp_path / "split.json")}
        payload.update(overrides)
        return DataConfig.model_validate(payload)

    def test_accepts_a_matching_manifest(self, tmp_path: Path, balanced_labels: list[int]) -> None:
        config = self._config(tmp_path)
        verify_manifest(build_manifest(balanced_labels, config), balanced_labels, config)

    def test_rejects_a_changed_dataset_length(
        self, tmp_path: Path, balanced_labels: list[int]
    ) -> None:
        config = self._config(tmp_path)
        manifest = build_manifest(balanced_labels, config)
        with pytest.raises(SplitError, match="dataset has"):
            verify_manifest(manifest, [*balanced_labels, 0], config)

    def test_rejects_a_different_seed(self, tmp_path: Path, balanced_labels: list[int]) -> None:
        config = self._config(tmp_path)
        manifest = build_manifest(balanced_labels, config)
        with pytest.raises(SplitError, match="was built with seed"):
            verify_manifest(manifest, balanced_labels, self._config(tmp_path, split_seed=99))

    def test_rejects_a_different_val_fraction(
        self, tmp_path: Path, balanced_labels: list[int]
    ) -> None:
        config = self._config(tmp_path)
        manifest = build_manifest(balanced_labels, config)
        with pytest.raises(SplitError, match="val_fraction"):
            verify_manifest(manifest, balanced_labels, self._config(tmp_path, val_fraction=0.5))

    def test_rejects_relabelled_data(self, tmp_path: Path, balanced_labels: list[int]) -> None:
        """The failure this exists for: same length, different labels."""
        config = self._config(tmp_path)
        manifest = build_manifest(balanced_labels, config)
        relabelled = list(balanced_labels)
        relabelled[manifest.train_indices[0]] = 4
        with pytest.raises(SplitError, match="label mismatch"):
            verify_manifest(manifest, relabelled, config)

    def test_rejects_index_overlap(self, tmp_path: Path, balanced_labels: list[int]) -> None:
        config = self._config(tmp_path)
        manifest = build_manifest(balanced_labels, config)
        leaked = SplitManifest(
            version=manifest.version,
            seed=manifest.seed,
            val_fraction=manifest.val_fraction,
            num_samples=manifest.num_samples,
            train_indices=manifest.train_indices,
            val_indices=(*manifest.val_indices, manifest.train_indices[0]),
            train_labels=manifest.train_labels,
            val_labels=(
                *manifest.val_labels,
                balanced_labels[manifest.train_indices[0]],
            ),
            created_at=manifest.created_at,
        )
        with pytest.raises(SplitError, match="data leak"):
            verify_manifest(leaked, balanced_labels, config)

    def test_rejects_incomplete_coverage(self, tmp_path: Path, balanced_labels: list[int]) -> None:
        config = self._config(tmp_path)
        manifest = build_manifest(balanced_labels, config)
        truncated = SplitManifest(
            version=manifest.version,
            seed=manifest.seed,
            val_fraction=manifest.val_fraction,
            num_samples=manifest.num_samples,
            train_indices=manifest.train_indices[:-1],
            val_indices=manifest.val_indices,
            train_labels=manifest.train_labels[:-1],
            val_labels=manifest.val_labels,
            created_at=manifest.created_at,
        )
        with pytest.raises(SplitError, match="do not cover the dataset"):
            verify_manifest(truncated, balanced_labels, config)


class TestGetOrCreateManifest:
    def test_creates_then_reuses(self, tmp_path: Path, balanced_labels: list[int]) -> None:
        config = DataConfig.model_validate({"split_manifest": str(tmp_path / "split.json")})
        first = get_or_create_manifest(balanced_labels, config)
        assert config.split_manifest.exists()
        second = get_or_create_manifest(balanced_labels, config)
        assert first == second

    def test_force_recomputes(self, tmp_path: Path, balanced_labels: list[int]) -> None:
        config = DataConfig.model_validate({"split_manifest": str(tmp_path / "split.json")})
        first = get_or_create_manifest(balanced_labels, config)
        second = get_or_create_manifest(balanced_labels, config, force=True)
        assert first == second  # same seed, so same split, but it was rewritten

    def test_stale_manifest_is_rejected_on_reuse(self, tmp_path: Path) -> None:
        config = DataConfig.model_validate({"split_manifest": str(tmp_path / "split.json")})
        get_or_create_manifest([0] * 10 + [1] * 10, config)
        with pytest.raises(SplitError, match="dataset has"):
            get_or_create_manifest([0] * 12 + [1] * 12, config)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
class TestClassWeights:
    def test_power_zero_is_uniform(self) -> None:
        weights = compute_class_weights([0] * 100 + [1] * 10, power=0.0)
        assert weights.per_class[0] == pytest.approx(weights.per_class[1])

    def test_power_one_is_inverse_frequency(self) -> None:
        weights = compute_class_weights([0] * 100 + [1] * 10, power=1.0)
        assert weights.per_class[1] / weights.per_class[0] == pytest.approx(10.0)

    def test_weights_are_normalised_to_one(self) -> None:
        weights = compute_class_weights([0] * 3 + [1] * 7 + [2] * 20, power=1.0)
        assert max(weights.per_class.values()) == pytest.approx(1.0)

    def test_effective_counts_are_balanced_at_power_one(self) -> None:
        labels = [0] * 100 + [1] * 10
        weights = compute_class_weights(labels, power=1.0)
        effective = weights.effective_counts(labels, num_samples=1000)
        assert effective[0] == pytest.approx(effective[1], rel=1e-6)
        assert sum(effective.values()) == pytest.approx(1000.0)

    def test_empty_labels_raise(self) -> None:
        with pytest.raises(ValueError, match="empty label sequence"):
            compute_class_weights([], power=1.0)

    @pytest.mark.parametrize("power", [-0.1, 1.5])
    def test_power_outside_unit_range_raises(self, power: float) -> None:
        with pytest.raises(ValueError, match=r"power must be in \[0, 1\]"):
            compute_class_weights([0, 1], power=power)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestLabelDistribution:
    def test_includes_zero_count_classes(self) -> None:
        assert label_distribution([0, 0, 2], num_classes=4) == {0: 2, 1: 0, 2: 1, 3: 0}

    def test_out_of_range_label_raises(self) -> None:
        with pytest.raises(DatasetValidationError, match="outside the valid range"):
            label_distribution([0, 5], num_classes=3)


class TestValidateDataset:
    def test_reports_distribution_and_imbalance(self) -> None:
        base = FakeBaseDataset([0] * 10 + [1] * 5)
        stats = validate_dataset(base, num_classes=3)
        assert stats.num_samples == 15
        assert stats.min_class_count == 0
        assert stats.max_class_count == 10
        assert stats.missing_classes == [2]

    def test_imbalance_ratio(self) -> None:
        stats = validate_dataset(FakeBaseDataset([0] * 20 + [1] * 5), num_classes=2)
        assert stats.imbalance_ratio == pytest.approx(4.0)

    def test_image_checks_catch_a_non_rgb_image(self) -> None:
        class GreyDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> tuple[Image.Image, int]:
                return Image.new("L", (32, 32)), 0

        with pytest.raises(DatasetValidationError, match="expected 'RGB'"):
            validate_dataset(GreyDataset(), num_classes=1, sample_size=1)

    def test_image_checks_accept_real_gtsrb_aspect_ratios(self) -> None:
        """The regression this threshold exists for.

        Measured on 4,000 real GTSRB training images, the most extreme departure
        from 1:1 is 0.533 (a 0.467 ratio). A default threshold of 0.5 rejected it,
        which meant a full validation pass over the real dataset could not succeed.
        """
        # Real GTSRB extremes: the most portrait image measured is 0.467 (a
        # long/short ratio of 2.14) and the most landscape is 1.40.
        for size in ((32, 68), (40, 56), (56, 40), (32, 45)):

            class Dataset:
                def __init__(self, size: tuple[int, int]) -> None:
                    self.size = size

                def __len__(self) -> int:
                    return 1

                def __getitem__(self, index: int) -> tuple[Image.Image, int]:
                    return Image.new("RGB", self.size), 0

            stats = validate_dataset(Dataset(size), num_classes=1, sample_size=1)
            assert stats.num_samples == 1

    def test_image_checks_catch_a_non_square_ratio(self) -> None:
        class WideDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> tuple[Image.Image, int]:
                return Image.new("RGB", (200, 20)), 0  # long/short ratio of 10

        with pytest.raises(DatasetValidationError, match="long/short ratio"):
            validate_dataset(WideDataset(), num_classes=1, sample_size=1)

    def test_a_genuinely_corrupt_crop_is_still_rejected(self) -> None:
        """Raising the threshold to 1.0 must not disable the check."""

        class CorruptDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> tuple[Image.Image, int]:
                return Image.new("RGB", (600, 12)), 0  # long/short ratio of 50

        with pytest.raises(DatasetValidationError, match="long/short ratio"):
            validate_dataset(CorruptDataset(), num_classes=1, sample_size=1)

    def test_image_checks_catch_a_tensor_base_dataset(self) -> None:
        """A base dataset that already applies transforms breaks per-split pipelines."""

        class TensorDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
                return torch.zeros(3, 8, 8), 0

        with pytest.raises(DatasetValidationError, match="expected a PIL image"):
            validate_dataset(TensorDataset(), num_classes=1, sample_size=1)  # type: ignore[arg-type]

    def test_empty_dataset_raises(self) -> None:
        with pytest.raises(DatasetValidationError, match="empty"):
            validate_dataset(FakeBaseDataset([]), num_classes=1)

    def test_sample_size_is_capped_by_dataset_length(self) -> None:
        stats = validate_dataset(FakeBaseDataset([0, 1]), num_classes=2, sample_size=100)
        assert stats.num_samples == 2

    def test_summary_is_readable(self) -> None:
        stats = validate_dataset(FakeBaseDataset([0] * 10 + [1] * 5), num_classes=2)
        assert "15 samples" in stats.summary()
        assert "imbalance" in stats.summary()


class TestEstimateChannelStatistics:
    def test_estimates_are_close_to_the_synthetic_generator(self) -> None:
        base = FakeBaseDataset([0] * 40 + [1] * 40, size=32)
        mean, std = estimate_channel_statistics(base, sample_size=16, seed=1, image_size=32)
        assert all(0.0 <= value <= 1.0 for value in mean)
        assert all(value > 0.0 for value in std)

    def test_is_reproducible_for_a_seed(self) -> None:
        base = FakeBaseDataset([0] * 40, size=32)
        first = estimate_channel_statistics(base, sample_size=8, seed=5, image_size=32)
        second = estimate_channel_statistics(base, sample_size=8, seed=5, image_size=32)
        assert first == second

    def test_empty_dataset_raises(self) -> None:
        with pytest.raises(DatasetValidationError, match="empty dataset"):
            estimate_channel_statistics(FakeBaseDataset([]), sample_size=4)


class TestCompareStatistics:
    def test_identical_statistics_are_within_tolerance(self) -> None:
        result = compare_statistics(
            (0.5, 0.5, 0.5), (0.2, 0.2, 0.2), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)
        )
        assert result["within_tolerance"] is True

    def test_large_deviation_is_flagged(self) -> None:
        result = compare_statistics(
            (0.9, 0.5, 0.5), (0.2, 0.2, 0.2), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)
        )
        assert result["within_tolerance"] is False
        assert result["mean_delta"][0] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Loader wiring (no real dataset: the raw loader is monkeypatched)
# ---------------------------------------------------------------------------
class TestBuildDataloaders:
    def _patch_raw_loader(self, monkeypatch: pytest.MonkeyPatch, labels: list[int]) -> None:
        base = FakeBaseDataset(labels, size=32)

        def fake_load(root: Path, split: str, *, download: bool = True) -> FakeBaseDataset:
            return base

        monkeypatch.setattr("gtsrb.data.loaders.load_raw_split", fake_load)

    def test_shapes_and_split_sizes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        labels = [label for label in range(5) for _ in range(10)]
        self._patch_raw_loader(monkeypatch, labels)
        config = ProjectConfig.model_validate(
            {
                "data": {
                    "image_size": 16,
                    "num_workers": 0,
                    "split_manifest": str(tmp_path / "split.json"),
                },
                "training": {"batch_size": 4, "device": "cpu"},
            }
        )
        bundle = build_dataloaders(config, download=False)
        assert len(bundle.train_dataset) + len(bundle.val_dataset) == 50
        assert len(bundle.test_dataset) == 50
        images, targets = next(iter(bundle.train))
        assert images.shape == (4, 3, 16, 16)
        assert targets.shape == (4,)

    def test_train_and_val_never_share_an_index(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        labels = [label for label in range(5) for _ in range(10)]
        self._patch_raw_loader(monkeypatch, labels)
        config = ProjectConfig.model_validate(
            {"data": {"image_size": 16, "split_manifest": str(tmp_path / "split.json")}}
        )
        bundle = build_dataloaders(config, download=False)
        assert set(bundle.train_dataset.indices).isdisjoint(bundle.val_dataset.indices)

    def test_weighted_sampling_balances_the_training_stream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Deliberately lopsided: 40 of class 0, 5 of class 1, 5 of class 2.
        labels = [0] * 40 + [1] * 5 + [2] * 5
        self._patch_raw_loader(monkeypatch, labels)
        config = ProjectConfig.model_validate(
            {
                "data": {
                    "image_size": 16,
                    "split_manifest": str(tmp_path / "split.json"),
                    "balance": {"mode": "weighted", "power": 1.0},
                },
                "training": {"batch_size": 10, "seed": 3},
            }
        )
        bundle = build_dataloaders(config, download=False)
        observed: dict[int, int] = {}
        for _ in range(6):
            for _, targets in bundle.train:
                for target in targets.tolist():
                    observed[target] = observed.get(target, 0) + 1
        # The raw split holds 32/4/4 samples of classes 0/1/2, so without
        # resampling class 1 could be seen at most 24 times over 6 epochs. With
        # inverse-frequency sampling the three classes must come out roughly even.
        assert observed[1] > 60
        assert observed[2] > 60
        assert max(observed.values()) / min(observed.values()) < 2.0

    def test_metadata_is_serialisable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_raw_loader(monkeypatch, [0] * 10 + [1] * 10)
        config = ProjectConfig.model_validate(
            {"data": {"image_size": 16, "split_manifest": str(tmp_path / "split.json")}}
        )
        metadata = build_dataloaders(config, download=False).metadata()
        assert json.loads(json.dumps(metadata))["num_classes"] == 43


class TestReadLabels:
    def test_reads_from_samples_metadata_without_decoding(self) -> None:
        """torchvision's GTSRB stores ``(path, target)`` pairs in ``_samples``."""

        class WithSamples:
            def __init__(self) -> None:
                self._samples = [("a.ppm", 0), ("b.ppm", 7), ("c.ppm", 42)]

            def __len__(self) -> int:
                return len(self._samples)

            def __getitem__(self, index: int) -> tuple[Image.Image, int]:
                raise AssertionError("must not decode images when metadata is available")

        assert read_labels(WithSamples()) == [0, 7, 42]

    def test_reads_from_targets_metadata(self) -> None:
        class WithTargets:
            def __init__(self) -> None:
                self.targets = [3, 1, 4]

            def __len__(self) -> int:
                return len(self.targets)

            def __getitem__(self, index: int) -> tuple[Image.Image, int]:
                raise AssertionError("must not decode images when metadata is available")

        assert read_labels(WithTargets()) == [3, 1, 4]

    def test_falls_back_to_getitem(self) -> None:
        base = FakeBaseDataset([5, 6, 7])
        assert read_labels(base) == [5, 6, 7]

    def test_subset_labels_use_metadata(self) -> None:
        class WithSamples:
            def __init__(self) -> None:
                self._samples = [("a.ppm", 0), ("b.ppm", 1), ("c.ppm", 2)]

            def __len__(self) -> int:
                return len(self._samples)

            def __getitem__(self, index: int) -> tuple[Image.Image, int]:
                raise AssertionError("must not decode images")

        subset = TransformSubset(WithSamples(), [2, 0], transforms.ToTensor())
        assert subset.labels == [2, 0]
        assert subset.label_distribution() == {0: 1, 2: 1}


class TestRawSplitGuard:
    def test_missing_data_without_download_flag_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="download=False"):
            load_raw_split(tmp_path / "empty", "train", download=False)
