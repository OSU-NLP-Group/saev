import time
import types

import psutil

from saev.utils.monitoring import DataloaderMonitor


class _StubReservoir:
    def __init__(self, fill_value: float) -> None:
        self._fill_value = fill_value

    def fill(self) -> float:
        return self._fill_value

    def set_fill(self, value: float) -> None:
        self._fill_value = value


class _StubLoader:
    def __init__(self, reservoir: _StubReservoir, manager_pid: int = -1) -> None:
        self.reservoir = reservoir
        self._manager_pid = manager_pid

    @property
    def manager_pid(self) -> int:
        return self._manager_pid

    def set_manager_pid(self, pid: int) -> None:
        self._manager_pid = pid


class _StubProcess:
    def __init__(
        self,
        pid: int,
        *,
        read_bytes: int = 0,
        cpu_percent: float = 0.0,
        io_exc: Exception | None = None,
        cpu_exc: Exception | None = None,
        running_exc: Exception | None = None,
        running: bool = True,
        children: list["_StubProcess"] | None = None,
        children_exc: Exception | None = None,
    ) -> None:
        self.pid = pid
        self._read_bytes = read_bytes
        self._cpu_percent = cpu_percent
        self._io_exc = io_exc
        self._cpu_exc = cpu_exc
        self._running_exc = running_exc
        self._running = running
        self._children = children or []
        self._children_exc = children_exc

    def io_counters(self) -> types.SimpleNamespace:
        if self._io_exc is not None:
            raise self._io_exc
        return types.SimpleNamespace(read_bytes=self._read_bytes)

    def set_read_bytes(self, value: int) -> None:
        self._read_bytes = value

    def cpu_percent(self, interval: float | None) -> float:
        if self._cpu_exc is not None:
            raise self._cpu_exc
        return self._cpu_percent

    def set_cpu_percent(self, value: float) -> None:
        self._cpu_percent = value

    def children(self, recursive: bool) -> list["_StubProcess"]:
        if self._children_exc is not None:
            raise self._children_exc
        return self._children

    def set_children(self, children: list["_StubProcess"]) -> None:
        self._children = children

    def is_running(self) -> bool:
        if self._running_exc is not None:
            raise self._running_exc
        return self._running

    def set_running(self, value: bool) -> None:
        self._running = value


def test_monitor_returns_buffer_when_manager_missing():
    loader = _StubLoader(_StubReservoir(0.4), manager_pid=-1)
    monitor = DataloaderMonitor(loader)
    metrics = monitor.compute()
    assert metrics == {"loader/buffer_fill": 0.4}


def test_monitor_preserves_warnings_when_manager_missing():
    loader = _StubLoader(_StubReservoir(0.1), manager_pid=-1)
    monitor = DataloaderMonitor(loader)
    monitor.warned_cpu = True
    monitor.warned_io = True
    monitor.can_read_cpu = False
    metrics = monitor.compute()
    assert metrics == {"loader/buffer_fill": 0.1}
    assert monitor.warned_cpu is True
    assert monitor.warned_io is True


def test_monitor_tracks_io_and_cpu_across_steps():
    reservoir = _StubReservoir(0.5)
    loader = _StubLoader(reservoir, manager_pid=123)

    child = _StubProcess(pid=124, cpu_percent=5.0)
    parent = _StubProcess(pid=123, read_bytes=1024, cpu_percent=7.5, children=[child])
    processes = {123: parent}

    def _factory(pid: int) -> _StubProcess:
        return processes[pid]

    monitor = DataloaderMonitor(loader, process_factory=_factory)

    metrics_first = monitor.compute(now=time.time())
    assert metrics_first["loader/buffer_fill"] == 0.5
    assert metrics_first["loader/read_mb"] == 0.0
    assert metrics_first["loader/read_mb_s"] == 0.0
    assert metrics_first["loader/cpu_util"] == 12.5

    parent.set_read_bytes(3072)
    parent.set_cpu_percent(10.0)
    child.set_cpu_percent(6.0)
    reservoir.set_fill(0.6)

    metrics_second = monitor.compute(now=time.time() + 2.0)
    assert metrics_second["loader/buffer_fill"] == 0.6
    assert metrics_second["loader/read_mb"] > 0.0
    assert metrics_second["loader/read_mb_s"] > 0.0
    assert metrics_second["loader/cpu_util"] == 16.0


def test_monitor_disables_io_on_access_denied():
    loader = _StubLoader(_StubReservoir(0.0), manager_pid=5)
    process = _StubProcess(
        pid=5, io_exc=psutil.AccessDenied(pid=5, name="io"), cpu_percent=1.0
    )

    def _factory(pid: int) -> _StubProcess:
        return process

    monitor = DataloaderMonitor(loader, process_factory=_factory)
    metrics = monitor.compute(now=time.time())
    assert "loader/read_mb" not in metrics
    assert monitor.can_read_io is False
    assert monitor.warned_io is True


def test_monitor_disables_cpu_on_access_denied():
    loader = _StubLoader(_StubReservoir(0.2), manager_pid=9)
    process = _StubProcess(
        pid=9, cpu_exc=psutil.AccessDenied(pid=9, name="cpu"), read_bytes=0
    )

    def _factory(pid: int) -> _StubProcess:
        return process

    monitor = DataloaderMonitor(loader, process_factory=_factory)
    metrics = monitor.compute(now=time.time())
    assert "loader/cpu_util" not in metrics
    assert monitor.can_read_cpu is False
    assert monitor.warned_cpu is True


def test_monitor_handles_children_errors():
    loader = _StubLoader(_StubReservoir(0.1), manager_pid=11)
    process = _StubProcess(
        pid=11,
        cpu_percent=0.0,
        children_exc=psutil.AccessDenied(pid=11, name="children"),
    )

    def _factory(pid: int) -> _StubProcess:
        return process

    monitor = DataloaderMonitor(loader, process_factory=_factory)
    _ = monitor.compute(now=time.time())
    assert monitor.children == []


def test_monitor_attach_resets_state():
    reservoir = _StubReservoir(0.3)
    loader = _StubLoader(reservoir, manager_pid=13)
    process = _StubProcess(pid=13, read_bytes=512, cpu_percent=2.0)

    def _factory(pid: int) -> _StubProcess:
        return process

    monitor = DataloaderMonitor(loader, process_factory=_factory)
    _ = monitor.compute(now=time.time())
    assert monitor.current_pid == 13
    assert monitor.process is process

    new_loader = _StubLoader(reservoir, manager_pid=14)
    monitor.attach(new_loader)
    assert monitor.current_pid is None
    assert monitor.process is None
    assert monitor.warned_cpu is False
    assert monitor.warned_io is False


def test_monitor_attach_noop_for_same_loader():
    loader = _StubLoader(_StubReservoir(0.2), manager_pid=1)
    monitor = DataloaderMonitor(loader)
    monitor.warned_cpu = True
    monitor.attach(loader)
    assert monitor.warned_cpu is True


def test_monitor_process_factory_failure():
    loader = _StubLoader(_StubReservoir(0.4), manager_pid=21)

    def _factory(pid: int) -> _StubProcess:
        raise psutil.NoSuchProcess(pid=pid, name="missing")

    monitor = DataloaderMonitor(loader, process_factory=_factory)
    metrics = monitor.compute(now=time.time())
    assert metrics == {"loader/buffer_fill": 0.4}


def test_monitor_manager_pid_callable_failure():
    class _Loader:
        def __init__(self) -> None:
            self.reservoir = _StubReservoir(0.1)

        def manager_pid(self) -> int:
            raise RuntimeError("boom")

    loader = _Loader()
    monitor = DataloaderMonitor(loader, process_factory=lambda _: None)  # type: ignore[arg-type]
    metrics = monitor.compute(now=time.time())
    assert metrics == {"loader/buffer_fill": 0.1}


def test_monitor_manager_pid_cast_failure():
    class _Loader:
        def __init__(self) -> None:
            self.reservoir = _StubReservoir(0.2)
            self.manager_pid = "not-an-int"

    loader = _Loader()
    monitor = DataloaderMonitor(loader, process_factory=lambda _: None)  # type: ignore[arg-type]
    metrics = monitor.compute(now=time.time())
    assert metrics == {"loader/buffer_fill": 0.2}


def test_monitor_reservoir_fill_exception():
    class _Reservoir:
        def fill(self) -> float:
            raise RuntimeError("fill failed")

    class _Loader:
        def __init__(self) -> None:
            self.reservoir = _Reservoir()
            self.manager_pid = -1

    loader = _Loader()
    monitor = DataloaderMonitor(loader, process_factory=lambda _: None)  # type: ignore[arg-type]
    metrics = monitor.compute(now=time.time())
    assert metrics == {"loader/buffer_fill": 0.0}


def test_monitor_is_running_exception() -> None:
    process = _StubProcess(pid=1, running_exc=RuntimeError("boom"))
    assert DataloaderMonitor._is_running.__wrapped__(process) is False


def test_monitor_is_running_missing_method() -> None:
    class _Bare:
        pass

    assert DataloaderMonitor._is_running.__wrapped__(_Bare()) is True


def test_monitor_is_running_nosuchprocess() -> None:
    process = _StubProcess(pid=2, running_exc=psutil.NoSuchProcess(pid=2, name="p"))
    assert DataloaderMonitor._is_running.__wrapped__(process) is False


def test_monitor_read_bytes_missing_method() -> None:
    loader = _StubLoader(_StubReservoir(0.1), manager_pid=5)

    class _Process:
        pid = 5

    monitor = DataloaderMonitor(loader, process_factory=lambda _: _Process())  # type: ignore[arg-type]
    metrics = monitor.compute(now=time.time())
    assert "loader/read_mb" not in metrics


def test_monitor_read_cpu_percent_missing_method() -> None:
    loader = _StubLoader(_StubReservoir(0.1), manager_pid=7)

    class _Process:
        pid = 7

    monitor = DataloaderMonitor(loader, process_factory=lambda _: _Process())  # type: ignore[arg-type]
    metrics = monitor.compute(now=time.time())
    assert metrics["loader/cpu_util"] == 0.0


def test_monitor_read_bytes_generic_exception() -> None:
    loader = _StubLoader(_StubReservoir(0.1), manager_pid=17)

    class _Process:
        pid = 17

        def io_counters(self):
            raise RuntimeError("io boom")

    monitor = DataloaderMonitor(loader, process_factory=lambda _: _Process())  # type: ignore[arg-type]
    metrics = monitor.compute(now=time.time())
    assert "loader/read_mb" not in metrics
    assert monitor.can_read_io is False


def test_monitor_read_bytes_missing_read_bytes_field() -> None:
    monitor = DataloaderMonitor(_StubLoader(_StubReservoir(0.1), manager_pid=23))

    class _Process:
        pid = 23

        def io_counters(self):
            return types.SimpleNamespace()

    result = DataloaderMonitor._read_bytes.__wrapped__(monitor, _Process(), time.time())
    assert result is None


def test_monitor_update_children_generic_exception() -> None:
    loader = _StubLoader(_StubReservoir(0.2), manager_pid=41)

    class _BoomProcess:
        pid = 41

        def children(self, recursive: bool) -> list[object]:
            raise RuntimeError("boom")

        def io_counters(self):
            return types.SimpleNamespace(read_bytes=0)

        def cpu_percent(self, interval: float | None) -> float:
            return 0.0

    monitor = DataloaderMonitor(loader, process_factory=lambda _: _BoomProcess())  # type: ignore[arg-type]
    metrics = monitor.compute(now=time.time())
    assert "loader/cpu_util" in metrics
    assert monitor.children == []


def test_monitor_read_cpu_percent_generic_exception() -> None:
    loader = _StubLoader(_StubReservoir(0.2), manager_pid=19)

    class _Process:
        pid = 19

        def cpu_percent(self, interval: float | None) -> float:
            raise RuntimeError("cpu boom")

        def io_counters(self):
            return types.SimpleNamespace(read_bytes=0)

    monitor = DataloaderMonitor(loader, process_factory=lambda _: _Process())  # type: ignore[arg-type]
    metrics = monitor.compute(now=time.time())
    assert "loader/cpu_util" not in metrics
    assert monitor.can_read_cpu is False


def test_monitor_read_cpu_percent_non_numeric() -> None:
    loader = _StubLoader(_StubReservoir(0.2), manager_pid=43)

    class _Process:
        pid = 43

        def cpu_percent(self, interval: float | None):
            return "oops"

        def io_counters(self):
            return types.SimpleNamespace(read_bytes=0)

    monitor = DataloaderMonitor(loader, process_factory=lambda _: _Process())  # type: ignore[arg-type]
    metrics = monitor.compute(now=time.time())
    assert metrics["loader/cpu_util"] == 0.0


def test_monitor_read_cpu_percent_nosuchprocess() -> None:
    loader = _StubLoader(_StubReservoir(0.2), manager_pid=29)

    class _Process:
        pid = 29

        def cpu_percent(self, interval: float | None) -> float:
            raise psutil.NoSuchProcess(pid=29, name="parent")

        def io_counters(self):
            return types.SimpleNamespace(read_bytes=0)

    monitor = DataloaderMonitor(loader, process_factory=lambda _: _Process())  # type: ignore[arg-type]
    metrics = monitor.compute(now=time.time())
    assert "loader/cpu_util" not in metrics
    assert monitor.can_read_cpu is False


def test_monitor_get_reservoir_fill_missing_reservoir():
    class _Loader:
        manager_pid = -1
        reservoir = None

    loader = _Loader()
    monitor = DataloaderMonitor(loader, process_factory=lambda _: None)  # type: ignore[arg-type]
    metrics = monitor.compute()
    assert metrics == {"loader/buffer_fill": 0.0}


def test_monitor_get_manager_pid_none():
    class _Loader:
        def __init__(self) -> None:
            self.reservoir = _StubReservoir(0.1)
            self.manager_pid = None

    loader = _Loader()
    monitor = DataloaderMonitor(loader, process_factory=lambda _: None)  # type: ignore[arg-type]
    metrics = monitor.compute()
    assert metrics == {"loader/buffer_fill": 0.1}


def test_monitor_can_read_cpu_false_branch():
    reservoir = _StubReservoir(0.3)
    loader = _StubLoader(reservoir, manager_pid=31)
    process = _StubProcess(pid=31, read_bytes=0, cpu_percent=0.0)

    def _factory(pid: int) -> _StubProcess:
        return process

    monitor = DataloaderMonitor(loader, process_factory=_factory)
    _ = monitor.compute(now=time.time())
    monitor.can_read_cpu = False
    monitor.warned_cpu = False
    _ = monitor.compute(now=time.time())
    assert monitor.warned_cpu is True
