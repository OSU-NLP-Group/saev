import time
import types

import psutil

from saev.utils.monitoring import LoaderMonitor


class _StubProcess:
    def __init__(
        self,
        pid: int,
        io_exc: Exception | None,
        read_bytes: int,
        cpu_percent_value: float,
    ) -> None:
        self.pid = pid
        self._io_exc = io_exc
        self._read_bytes = read_bytes
        self._cpu_percent_value = cpu_percent_value

    def io_counters(self) -> types.SimpleNamespace:
        if self._io_exc is not None:
            raise self._io_exc
        return types.SimpleNamespace(read_bytes=self._read_bytes)

    def cpu_percent(self, interval: float | None) -> float:
        return self._cpu_percent_value

    def set_read_bytes(self, value: int) -> None:
        self._read_bytes = value


def test_loader_monitor_resets_when_pid_changes():
    now = time.time()
    monitor = LoaderMonitor()

    failing_proc = _StubProcess(
        pid=123,
        io_exc=psutil.AccessDenied(pid=123, name="stub"),
        read_bytes=1024,
        cpu_percent_value=5.0,
    )

    metrics = LoaderMonitor.collect.__wrapped__(
        monitor,
        p_dataloader=failing_proc,
        p_children=[],
        reservoir_fill=0.5,
        now=now,
    )

    assert metrics["loader/buffer_fill"] == 0.5
    assert "loader/read_mb" not in metrics
    assert "loader/read_mb_s" not in metrics
    assert monitor.can_read_io is False
    assert monitor.warned_io is True

    healthy_proc = _StubProcess(
        pid=456,
        io_exc=None,
        read_bytes=2048,
        cpu_percent_value=7.5,
    )

    metrics_after_restart = LoaderMonitor.collect.__wrapped__(
        monitor,
        p_dataloader=healthy_proc,
        p_children=[],
        reservoir_fill=0.25,
        now=now + 1.0,
    )

    assert metrics_after_restart["loader/buffer_fill"] == 0.25
    assert metrics_after_restart["loader/read_mb"] == 0.0
    assert metrics_after_restart["loader/read_mb_s"] == 0.0
    assert monitor.can_read_io is True
    assert monitor.warned_io is False

    healthy_proc.set_read_bytes(4096)

    metrics_next = LoaderMonitor.collect.__wrapped__(
        monitor,
        p_dataloader=healthy_proc,
        p_children=[],
        reservoir_fill=0.3,
        now=now + 2.0,
    )

    assert metrics_next["loader/read_mb"] > 0.0
    assert metrics_next["loader/read_mb_s"] > 0.0
