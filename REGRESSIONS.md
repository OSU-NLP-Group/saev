# Regressions

Last checked: 2025-09-16

# 9 failing test(s)

- tests/test_nn_modeling.py::test_remove_parallel_grads_handles_non_normalized_rows
- tests/test_nn_objectives.py::test_safe_mse_hypothesis
- tests/test_reservoir_buffer.py::test_blocking_get_when_empty[proc]
- tests/test_reservoir_buffer.py::test_blocking_put_when_full[proc]
- tests/test_ring_buffer.py::test_blocking_get_when_empty[proc]
- tests/test_ring_buffer.py::test_blocking_put_when_full[proc]
- tests/test_ring_buffer.py::test_capacity_never_exceeded
- tests/test_writers_properties.py::test_shard_size_consistency
- tests/test_writers_properties.py::test_shard_writer_and_dataset_e2e

# Coverage

Coverage: 1991/2864 lines (69.5%)
