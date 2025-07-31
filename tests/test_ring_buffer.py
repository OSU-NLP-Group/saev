import os
import queue
import signal
import threading
import time

import beartype
import pytest
import torch
import torch.multiprocessing as mp
from hypothesis import given, settings
from hypothesis import strategies as st

from saev.data.buffers import RingBuffer

mp.set_start_method("spawn", force=True)

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


@pytest.fixture(params=["thread", "proc"])
def backend(request):
    """
    Returns a namespace with .Event .Worker and .kwargs so the main test is agnostic to threading vs multiprocessing.
    """
    if request.param == "thread":

        @beartype.beartype
        class Namespace:  # lightweight namespace
            Event = threading.Event
            Worker = threading.Thread
            Queue = queue.Queue
            kwargs: dict = dict(daemon=True)

        return Namespace

    elif request.param == "proc":
        mp.set_start_method("spawn", force=True)  # safe to call repeatedly

        @beartype.beartype
        class Namespace:
            Event = mp.Event
            Worker = mp.Process
            Queue = mp.Queue
            kwargs: dict = dict(daemon=True)

        return Namespace
    else:
        raise ValueError(request.param)


@given(
    st.lists(
        st.integers(min_value=INT32_MIN, max_value=INT32_MAX), min_size=1, max_size=100
    )
)
def test_fifo(xs):
    r = RingBuffer(len(xs), (1,), dtype=torch.int32)
    for x in xs:
        r.put(torch.tensor([x], dtype=torch.int32))
    for x in xs:
        assert r.get().item() == x


def test_init_and_close():
    """Ring creates empty and survives multiple close() calls."""
    r = RingBuffer(slots=4, shape=(2,), dtype=torch.float32)
    assert r.qsize() == 0
    r.close()
    r.close()  # idempotent


def test_basic_put_get_order():
    """FIFO ordering for a short, known sequence."""
    seq = [11, 22, 33, 44]
    r = RingBuffer(slots=4, shape=(1,), dtype=torch.int32)

    for x in seq:
        r.put(torch.tensor([x], dtype=torch.int32))

    out = [r.get().item() for _ in seq]
    assert out == seq


@given(
    st.lists(
        st.integers(min_value=INT32_MIN, max_value=INT32_MAX), min_size=1, max_size=32
    )
)
def test_capacity_never_exceeded(xs):
    """
    Random round-trip: after each put+get the queue size
    never exceeds its declared capacity.
    """
    cap = 8
    r = RingBuffer(slots=cap, shape=(1,), dtype=torch.int32)

    for x in xs:
        r.put(torch.tensor([x], dtype=torch.int32))
        assert 0 < r.qsize() <= cap
        _ = r.get()
        assert r.qsize() <= cap


def test_exact_capacity_cycle():
    """Fill to capacity, drain to zero, repeat once."""
    cap = 4
    r = RingBuffer(slots=cap, shape=(1,), dtype=torch.int32)

    # fill
    for i in range(cap):
        r.put(torch.tensor([i], dtype=torch.int32))
    assert r.qsize() == cap

    # drain
    for i in range(cap):
        assert r.get()[0] == i
    assert r.qsize() == 0

    # second cycle to catch wrap-around bugs
    for i in range(cap):
        r.put(torch.tensor([i + 100], dtype=torch.int32))
    for i in range(cap):
        assert r.get()[0] == i + 100


@beartype.beartype
def _consume_blocking(ring: RingBuffer, started, finished):
    # blocks until producer frees slot
    started.set()
    _ = ring.get()
    finished.set()


@beartype.beartype
def _produce_blocking(ring: RingBuffer, started, finished, tensor):
    started.set()
    ring.put(tensor)
    finished.set()


@beartype.beartype
def _produce_n(ring: RingBuffer, n: int, start: int = 0):
    for i in range(n):
        ring.put(torch.tensor([i + start], dtype=torch.int32))


@beartype.beartype
def _collect_n(ring: RingBuffer, n: int, q):
    vals = []
    for _ in range(n):
        vals.append(ring.get().item())
    q.put(vals)


@beartype.beartype
def _consume_n(ring: RingBuffer, n: int):
    for _ in range(n):
        ring.get()


def test_blocking_put_when_full(backend):
    """
    put() must block when the buffer is full. We fill the ring, start a put in another process/thread, ensure it waits, then free a slot and verify the process/thread finishes.
    """
    cap = 16
    ring = RingBuffer(slots=cap, shape=(1,), dtype=torch.int32)

    for i in range(cap):
        ring.put(torch.tensor([i], dtype=torch.int32))  # buffer now full

    started = backend.Event()
    finished = backend.Event()

    w = backend.Worker(
        target=_produce_blocking,
        args=(ring, started, finished, torch.tensor([99], dtype=torch.int32)),
        **backend.kwargs,
    )
    w.start()

    started.wait(timeout=1)
    # producer should be blocked because buffer is full
    assert not finished.is_set()

    # free one slot -> producer should complete
    _ = ring.get()
    finished.wait(timeout=5)
    assert finished.is_set(), "put() did not unblock after space freed"


def test_blocking_get_when_empty(backend):
    """get() must block until an item is produced."""
    ring = RingBuffer(2, (1,), dtype=torch.int32)

    started = backend.Event()
    finished = backend.Event()

    w = backend.Worker(
        target=_consume_blocking, args=(ring, started, finished), **backend.kwargs
    )
    w.start()

    started.wait(1)
    assert not finished.is_set()

    # Put some arbitrary data; now _consume_blocking should pass.
    ring.put(torch.tensor([42], dtype=torch.int32))
    finished.wait(timeout=5)
    assert finished.is_set()


def test_wraparound_large_volume():
    """Push >> slots elements; order must still hold."""
    slots = 8
    total = 1_000
    ring = RingBuffer(slots, (1,), dtype=torch.int32)
    for i in range(total):
        ring.put(torch.tensor([i], dtype=torch.int32))
        out = ring.get()
        assert out.item() == i
    assert ring.qsize() == 0


def test_across_process_visibility(backend):
    """Producer in one process, consumer in another share the same data."""
    slots, total = 4, 10
    ring = RingBuffer(slots, (1,), dtype=torch.int32)

    q = backend.Queue()
    w1 = backend.Worker(target=_produce_n, args=(ring, total), **backend.kwargs)
    w2 = backend.Worker(target=_collect_n, args=(ring, total, q), **backend.kwargs)
    w1.start()
    w2.start()
    w1.join()
    w2.join()

    assert q.get() == list(range(10))


def test_put_shape_dtype_mismatch():
    rb = RingBuffer(2, (3,), dtype=torch.float32)
    with pytest.raises(ValueError):
        rb.put(torch.zeros(4, dtype=torch.float32))  # wrong shape
    with pytest.raises(ValueError):
        rb.put(torch.zeros(3, dtype=torch.int32))  # wrong dtype


def test_qsize_monotone_under_race(backend):
    """spawn two produces and two consumers; assert that qsize is never negative and never > capacity."""
    cap = 4
    ring = RingBuffer(cap, (1,), dtype=torch.int32)
    n = 200

    workers = [
        backend.Worker(target=_produce_n, args=(ring, n), **backend.kwargs)
        for _ in range(2)
    ] + [
        backend.Worker(target=_consume_n, args=(ring, n), **backend.kwargs)
        for _ in range(2)
    ]
    for w in workers:
        w.start()

    # sample qsize periodically
    for _ in range(50):
        qs = ring.qsize()
        assert 0 <= qs <= cap
        time.sleep(0.01)

    for w in workers:
        w.join()


def test_many_producers_consumers(backend):
    n_producers = 2
    slots = 8
    ring = RingBuffer(slots, (1,), torch.int32)
    per_proc = 100

    # Sum of arithmetic sequences
    expected = (
        per_proc * sum(range(n_producers))
        + n_producers * ((per_proc - 1) * (per_proc)) // 2
    )

    q = backend.Queue()
    producers = [
        backend.Worker(
            target=_produce_n,
            args=(ring, per_proc),
            kwargs=dict(start=i),
            **backend.kwargs,
        )
        for i in range(n_producers)
    ]
    consumers = [
        backend.Worker(target=_collect_n, args=(ring, per_proc, q), **backend.kwargs)
        for _ in range(2)
    ]
    for w in producers + consumers:
        w.start()
    for w in producers + consumers:
        w.join()
    collected = [q.get() for _ in consumers]
    assert sum([sum(c) for c in collected]) == expected


def _produce_then_die(ring: RingBuffer):
    ring.put(torch.tensor([1], dtype=torch.int32))
    os.kill(os.getpid(), signal.SIGKILL)


def test_producer_crash_does_not_corrupt():
    ring = RingBuffer(4, (1,), torch.int32)

    p = mp.Process(target=_produce_then_die, args=(ring,))
    p.start()
    p.join()

    # queue should still work
    ring.put(torch.tensor([2], dtype=torch.int32))
    assert ring.get().item() == 1
    assert ring.get().item() == 2


@settings(deadline=None, max_examples=20)
@given(
    st.lists(
        st.tuples(st.sampled_from(["put", "get"]), st.integers(0, 100)),
        min_size=1,
        max_size=200,
    )
)
def test_fuzz_interleaved_ops(seq):
    rb = RingBuffer(4, (1,), torch.int32)
    outstanding = 0
    for op, val in seq:
        if op == "put":
            if outstanding < 4:
                rb.put(torch.tensor([val], dtype=torch.int32))
                outstanding += 1
        elif op == "get":
            if outstanding:
                rb.get()
                outstanding -= 1
        else:
            raise ValueError(op)
        qs = rb.qsize()
        assert qs == outstanding
