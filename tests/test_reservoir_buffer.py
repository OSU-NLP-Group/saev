# tests/test_resevoir_buffer.py
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

from saev.data.buffers import ReservoirBuffer

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


def test_init_and_close():
    """Ring creates empty and survives multiple close() calls."""
    r = ReservoirBuffer(capacity=4, shape=(2,), dtype=torch.float32)
    assert r.qsize() == 0
    r.close()
    r.close()  # idempotent


def test_basic_put_get():
    """What goes in is what comes out (any ordering) for a short, known sequence."""
    seq = [11, 22, 33, 44]
    r = ReservoirBuffer(capacity=4, shape=(1,), dtype=torch.int32)

    for x in seq:
        r.put(torch.tensor([x], dtype=torch.int32))

    out = [r.get(1)[0].item() for _ in seq]
    assert set(out) == set(seq)


@settings(deadline=None, max_examples=100)
@given(st.integers(min_value=1, max_value=1000), st.data())
def test_all_put_get(capacity, data):
    """What goes in is what comes out (any ordering) for any sequence."""
    seq = data.draw(
        st.lists(
            st.integers(min_value=INT32_MIN, max_value=INT32_MAX),
            min_size=1,
            max_size=capacity,
        )
    )
    r = ReservoirBuffer(capacity=capacity, shape=(1,), dtype=torch.int32)

    for x in seq:
        r.put(torch.tensor([x], dtype=torch.int32))

    out = [r.get(1)[0].item() for _ in seq]
    assert set(out) == set(seq)


@settings(deadline=None, max_examples=200)
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
    capacity = 8
    r = ReservoirBuffer(capacity=capacity, shape=(1,), dtype=torch.int32)

    for x in xs:
        r.put(torch.tensor([x], dtype=torch.int32))
        assert 0 < r.qsize() <= capacity
        r.get(1)
        assert r.qsize() <= capacity


def test_exact_capacity_cycle():
    """Fill to capacity, drain to zero, repeat once."""
    capacity = 4
    r = ReservoirBuffer(capacity=capacity, shape=(1,), dtype=torch.int32)

    # fill
    for i in range(capacity):
        r.put(torch.tensor([i], dtype=torch.int32))
    assert r.qsize() == capacity

    # drain
    for i in range(capacity):
        assert r.get(1)[0].item() in set(range(capacity))
    assert r.qsize() == 0

    # second cycle to catch wrap-around bugs
    for i in range(capacity):
        r.put(torch.tensor([i + 100], dtype=torch.int32))
    for i in range(capacity):
        assert r.get(1)[0].item() in set(range(100, capacity + 100))


@beartype.beartype
def _consume_blocking(ring: ReservoirBuffer, started, finished):
    # blocks until producer frees slot
    started.set()
    ring.get(1)
    finished.set()


@beartype.beartype
def _produce_blocking(ring: ReservoirBuffer, started, finished, tensor):
    started.set()
    ring.put(tensor)
    finished.set()


@beartype.beartype
def _produce_n(ring: ReservoirBuffer, n: int, start: int = 0):
    for i in range(n):
        ring.put(torch.tensor([i + start], dtype=torch.int32))


@beartype.beartype
def _collect_n(ring: ReservoirBuffer, n: int, q):
    vals = []
    for _ in range(n):
        tensor, meta = ring.get(1)
        vals.append(tensor.item())
    q.put(vals)


@beartype.beartype
def _consume_n(ring: ReservoirBuffer, n: int):
    for _ in range(n):
        ring.get(1)


def test_blocking_put_when_full(backend):
    """
    put() must block when the buffer is full. We fill the ring, start a put in another process/thread, ensure it waits, then free a slot and verify the process/thread finishes.
    """
    capacity = 16
    ring = ReservoirBuffer(capacity=capacity, shape=(1,), dtype=torch.int32)

    for i in range(capacity):
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
    _ = ring.get(1)
    finished.wait(timeout=5)
    assert finished.is_set(), "put() did not unblock after space freed"


def test_blocking_get_when_empty(backend):
    """get() must block until an item is produced."""
    ring = ReservoirBuffer(2, (1,), dtype=torch.int32)

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


def test_across_process_visibility(backend):
    """Producer in one process, consumer in another share the same data."""
    slots, total = 4, 10
    ring = ReservoirBuffer(slots, (1,), dtype=torch.int32)

    q = backend.Queue()
    w1 = backend.Worker(target=_produce_n, args=(ring, total), **backend.kwargs)
    w2 = backend.Worker(target=_collect_n, args=(ring, total, q), **backend.kwargs)
    w1.start()
    w2.start()
    w1.join()
    w2.join()

    assert set(q.get()) == set(range(10))


def test_put_shape_dtype_mismatch():
    rb = ReservoirBuffer(2, (3,), dtype=torch.float32)
    with pytest.raises(ValueError):
        rb.put(torch.zeros(4, dtype=torch.float32))  # wrong shape
    with pytest.raises(ValueError):
        rb.put(torch.zeros(3, dtype=torch.int32))  # wrong dtype


def test_qsize_monotone_under_race(backend):
    """spawn two produces and two consumers; assert that qsize is never negative and never > capacity."""
    cap = 4
    ring = ReservoirBuffer(cap, (1,), dtype=torch.int32)
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
    ring = ReservoirBuffer(slots, (1,), dtype=torch.int32)
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


def _produce_then_die(ring: ReservoirBuffer):
    ring.put(torch.tensor([1], dtype=torch.int32))
    os.kill(os.getpid(), signal.SIGKILL)


def test_producer_crash_does_not_corrupt():
    ring = ReservoirBuffer(4, (1,), dtype=torch.int32)

    p = mp.Process(target=_produce_then_die, args=(ring,))
    p.start()
    p.join()

    # queue should still work
    ring.put(torch.tensor([2], dtype=torch.int32))
    tensor, metas = ring.get(2)
    assert set(tensor.squeeze().tolist()) == {1, 2}


@settings(deadline=None, max_examples=20)
@given(
    st.lists(
        st.tuples(st.sampled_from(["put", "get"]), st.integers(0, 100)),
        min_size=1,
        max_size=200,
    )
)
def test_fuzz_interleaved_ops(seq):
    rb = ReservoirBuffer(capacity=4, shape=(1,), dtype=torch.int32)
    outstanding = 0
    for op, val in seq:
        if op == "put":
            if outstanding < 4:
                rb.put(torch.tensor([val], dtype=torch.int32))
                outstanding += 1
        elif op == "get":
            if outstanding:
                rb.get(1)
                outstanding -= 1
        else:
            raise ValueError(op)
        qs = rb.qsize()
        assert qs == outstanding


@beartype.beartype
def _try_get(ring: ReservoirBuffer, bsz: int, timeout: float, q):
    """
    Attempt ring.get(); put ('ok', tensor) or ('timeout', None) in q.
    Runs in a thread or subprocess (supplied by the `backend` fixture).
    """
    try:
        tensor, _ = ring.get(bsz, timeout=timeout)
        q.put(("ok", tensor))
    except TimeoutError:
        q.put(("timeout", None))


@beartype.beartype
def _delayed_put(ring: ReservoirBuffer, tensors, delay: float):
    """
    Sleep `delay` seconds then put every tensor into the buffer.
    """
    time.sleep(delay)
    for t in tensors:
        ring.put(t)


def test_get_timeout_raises_and_recovers():
    """
    Calling get() for more items than are available with a finite timeout
    must raise TimeoutError *and* leave the internal state unchanged.
    """
    rb = ReservoirBuffer(4, (1,), dtype=torch.int32)
    rb.put(torch.tensor([7], dtype=torch.int32))  # only *one* item present

    with pytest.raises(TimeoutError):
        rb.get(2, timeout=0.15)  # asks for *two*

    # queue state intact – still exactly the original element present
    assert rb.qsize() == 1
    tensor, _ = rb.get(1, timeout=0.1)
    assert tensor.item() == 7


def test_partial_acquire_rollback():
    """
    When part‑way through a batch acquire, a timeout must roll back
    any semaphores already taken (no permit leak, no size skew).
    """
    rb = ReservoirBuffer(3, (1,), dtype=torch.int32)
    # two items in queue – one short of the requested batch
    rb.put(torch.tensor([1], dtype=torch.int32))
    rb.put(torch.tensor([2], dtype=torch.int32))

    start_size = rb.qsize()
    with pytest.raises(TimeoutError):
        rb.get(3, timeout=0.15)  # needs 3, gets stuck after 2

    # Size still identical; subsequent ops behave normally.
    assert rb.qsize() == start_size
    rb.put(torch.tensor([3], dtype=torch.int32))
    xs, _ = rb.get(3, timeout=0.1)
    assert set(xs.squeeze().tolist()) == {1, 2, 3}


def test_get_waits_until_items_arrive(backend):
    """
    A thread / process that calls get(..., timeout) should block until the batch
    becomes available *or* the timeout expires – whichever happens first.
    """
    rb = ReservoirBuffer(4, (1,), dtype=torch.int32)

    q = backend.Queue()
    worker = backend.Worker(target=_try_get, args=(rb, 2, 1.0, q), **backend.kwargs)
    worker.start()

    # ensure the consumer is definitely waiting on the first semaphore
    time.sleep(0.2)
    assert q.empty()

    # now deliver exactly the required batch well within the timeout
    producer = backend.Worker(
        target=_delayed_put,
        args=(
            rb,
            [
                torch.tensor([11], dtype=torch.int32),
                torch.tensor([22], dtype=torch.int32),
            ],
            0.1,  # wait a bit less than the remaining timeout
        ),
        **backend.kwargs,
    )
    producer.start()
    worker.join(timeout=2.0)
    producer.join(timeout=2.0)

    status, tensor = q.get()
    assert status == "ok"
    assert set(tensor.squeeze().tolist()) == {11, 22}
