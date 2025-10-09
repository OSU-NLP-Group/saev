# Regressions

Last checked: 2025-10-08

# 17 failing test(s)

- tests/test_nn_modeling.py::test_dump_load_roundtrip_exhaustive
- tests/test_nn_modeling.py::test_dump_load_roundtrip_hypothesis
- tests/test_nn_modeling.py::test_dump_load_roundtrip_simple[sae_cfg0]
- tests/test_nn_modeling.py::test_dump_load_roundtrip_simple[sae_cfg1]
- tests/test_nn_modeling.py::test_dump_load_roundtrip_simple[sae_cfg2]
- tests/test_nn_modeling.py::test_load_existing_checkpoint[osunlp/SAE_BioCLIP_24K_ViT-B-16_iNat21]
- tests/test_nn_modeling.py::test_load_existing_checkpoint[osunlp/SAE_CLIP_24K_ViT-B-16_IN1K]
- tests/test_nn_modeling.py::test_load_existing_checkpoint[osunlp/SAE_DINOv2_24K_ViT-B-14_IN1K]
- tests/test_nn_modeling.py::test_remove_parallel_grads_handles_non_normalized_rows
- tests/test_nn_objectives.py::test_safe_mse_hypothesis
- tests/test_ordered_dataloader.py::test_real_shards_sequential_order_with_labels[shards=1a2cdb29]
- tests/test_reservoir_buffer.py::test_blocking_get_when_empty[proc]
- tests/test_reservoir_buffer.py::test_blocking_put_when_full[proc]
- tests/test_ring_buffer.py::test_blocking_get_when_empty[proc]
- tests/test_ring_buffer.py::test_blocking_put_when_full[proc]
- tests/test_shards_properties.py::test_shards_json_is_emitted
- tests/test_unfold.py::test_hypothesis_special_values

# Coverage

Coverage: 1993/2940 lines (67.8%)
