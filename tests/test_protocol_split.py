import pytest
import torch

from evaluate_protocol import select_real_feature_subset


def test_final_protocol_subset_does_not_overlap_development_subset() -> None:
    features_by_class = [torch.arange(1000).reshape(500, 2) + class_id * 1000 for class_id in range(10)]
    development = select_real_feature_subset(
        features_by_class, 100, "development", 100, seed=2026
    )
    final = select_real_feature_subset(features_by_class, 100, "final", 100, seed=2026)
    later_final = select_real_feature_subset(
        features_by_class,
        100,
        "final",
        100,
        seed=2026,
        final_offset=100,
    )

    for development_class, final_class, later_class in zip(
        development, final, later_final
    ):
        development_ids = set(development_class[:, 0].tolist())
        final_ids = set(final_class[:, 0].tolist())
        later_ids = set(later_class[:, 0].tolist())
        assert development_ids.isdisjoint(final_ids)
        assert development_ids.isdisjoint(later_ids)
        assert final_ids.isdisjoint(later_ids)


def test_final_protocol_subset_rejects_overflow() -> None:
    features_by_class = [torch.zeros(10, 4)]

    with pytest.raises(ValueError, match="only 5 are available"):
        select_real_feature_subset(
            features_by_class,
            samples_per_class=6,
            partition="final",
            development_samples_per_class=5,
            seed=42,
        )


def test_real_reference_subset_is_disjoint_from_evaluated_slices() -> None:
    from evaluate_protocol import select_reference_subset

    features_by_class = [torch.arange(1000).reshape(500, 2) + class_id * 1000 for class_id in range(10)]
    final = select_real_feature_subset(
        features_by_class, 100, "final", 100, seed=2026, final_offset=100
    )
    reference = select_reference_subset(
        features_by_class, 100, seed=2026, used_end_index=300
    )

    for final_class, reference_class in zip(final, reference):
        assert set(final_class[:, 0].tolist()).isdisjoint(reference_class[:, 0].tolist())
    with pytest.raises(ValueError, match="disjoint"):
        select_reference_subset(features_by_class, 300, seed=2026, used_end_index=300)
