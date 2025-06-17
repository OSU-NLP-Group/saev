import threading

import beartype
import pytest
import torch
import torch.multiprocessing as mp
from hypothesis import given
from hypothesis import strategies as st

mp.set_start_method("spawn", force=True)

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

from saev.data.utils import RingBuffer


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
            kwargs: dict = dict(daemon=True)

        return Namespace

    elif request.param == "proc":
        mp.set_start_method("spawn", force=True)  # safe to call repeatedly

        @beartype.beartype
        class Namespace:
            Event = mp.Event
            Worker = mp.Process
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
def _produce_n(ring: RingBuffer, n: int):
    for i in range(n):
        ring.put(torch.tensor([i], dtype=torch.int32))


@beartype.beartype
def _consume_n(ring: RingBuffer, n: int, q):
    vals = []
    for _ in range(n):
        vals.append(ring.get().item())
    q.put(vals)


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


def test_across_process_visibility():
    """Producer in one process, consumer in another share the same data."""
    slots, total = 4, 10
    ring = RingBuffer(slots, (1,), dtype=torch.int32)

    q = mp.Queue()
    w1 = mp.Process(target=_produce_n, args=(ring, total))
    w2 = mp.Process(target=_consume_n, args=(ring, total, q))
    w1.start()
    w2.start()
    w1.join()
    w2.join()

    assert q.get() == list(range(10))
