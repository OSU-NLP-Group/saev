# Regressions

Last checked: 2025-10-03

# 34 failing test(s)

- tests/test_indexed_dataset.py::test_all_layers_not_implemented
- tests/test_indexed_dataset.py::test_invalid_layer
- tests/test_indexed_dataset.py::test_missing_shard_file_not_detected_at_init
- tests/test_indexed_dataset.py::test_nonexistent_shard_root
- tests/test_nn_modeling.py::test_load_existing_checkpoint[osunlp/SAE_BioCLIP_24K_ViT-B-16_iNat21]
- tests/test_nn_modeling.py::test_load_existing_checkpoint[osunlp/SAE_CLIP_24K_ViT-B-16_IN1K]
- tests/test_nn_modeling.py::test_load_existing_checkpoint[osunlp/SAE_DINOv2_24K_ViT-B-14_IN1K]
- tests/test_nn_modeling.py::test_remove_parallel_grads_handles_non_normalized_rows
- tests/test_nn_objectives.py::test_safe_mse_hypothesis
- tests/test_ordered_dataloader.py::test_missing_shard_file_not_detected_at_init
- tests/test_ordered_dataloader.py::test_no_patch_filtering_occurs
- tests/test_ordered_dataloader.py::test_ordered_dataloader_with_tiny_fake_dataset
- tests/test_ordered_dataloader.py::test_patch_labels_consistency_across_batches
- tests/test_ordered_dataloader.py::test_patch_labels_dtype_and_range
- tests/test_ordered_dataloader.py::test_patch_labels_not_returned_when_missing
- tests/test_ordered_dataloader.py::test_patch_labels_returned_when_available
- tests/test_ordered_dataloader.py::test_patch_labels_with_multiple_shards
- tests/test_ordered_dataloader.py::test_real_shards_label_distribution
- tests/test_ordered_dataloader.py::test_real_shards_no_filtering
- tests/test_ordered_dataloader.py::test_real_shards_reproducibility_with_labels
- tests/test_ordered_dataloader.py::test_real_shards_sequential_order_with_labels
- tests/test_ordered_dataloader.py::test_real_shards_with_labels
- tests/test_reservoir_buffer.py::test_blocking_get_when_empty[proc]
- tests/test_reservoir_buffer.py::test_blocking_put_when_full[proc]
- tests/test_ring_buffer.py::test_blocking_get_when_empty[proc]
- tests/test_ring_buffer.py::test_blocking_put_when_full[proc]
- tests/test_ring_buffer.py::test_capacity_never_exceeded
- tests/test_shards_disk.py::test_metadata_ex_per_shard_matches_disk
- tests/test_shards_disk.py::test_metadata_n_shards_matches_disk
- tests/test_shards_properties.py::test_shard_writer_and_dataset_e2e
- tests/test_shards_seg.py::test_labels_bin_with_cls_token
- tests/test_shuffled_dataloader.py::test_missing_shard_file_not_detected_at_init
- tests/test_shuffled_dataloader.py::test_no_child_leak
- tests/test_unfold.py::test_hypothesis_special_values

# Coverage

Coverage: 1773/2925 lines (60.6%)
