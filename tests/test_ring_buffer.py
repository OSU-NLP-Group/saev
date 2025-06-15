import threading

import torch
from hypothesis import given
from hypothesis import strategies as st

from saev.data.utils import RingBuffer


@given(st.lists(st.integers(), min_size=1, max_size=100))
def test_fifo(xs):
    r = RingBuffer(8, (1,), dtype=torch.int32)
    for x in xs:
        r.put(torch.tensor([x], dtype=torch.int32))
    for x in xs:
        assert r.get()[0] == x


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

    out = [r.get()[0] for _ in seq]
    assert out == seq


@given(st.lists(st.integers(), min_size=1, max_size=32))
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


def test_blocking_put_when_full():
    """
    put() must block when the buffer is full.
    We fill the ring, start a put in another thread, ensure
    it waits, then free a slot and verify the thread finishes.
    """
    cap = 2
    r = RingBuffer(slots=cap, shape=(1,), dtype=torch.int32)

    for i in range(cap):
        r.put(torch.tensor([i], dtype=torch.int32))  # buffer now full

    started = threading.Event()
    finished = threading.Event()

    def producer():
        started.set()
        r.put(torch.tensor([99], dtype=torch.int32))
        finished.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    started.wait(timeout=1)
    # producer should be blocked because buffer is full
    assert not finished.is_set()

    # free one slot -> producer should complete
    _ = r.get()
    finished.wait(timeout=1)
    assert finished.is_set(), "put() did not unblock after space freed"
