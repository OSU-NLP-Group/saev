import logging
import time
from collections.abc import Callable

import beartype
import psutil

logger = logging.getLogger(__name__)


@beartype.beartype
class DataloaderMonitor:
    """
    Tracks IO and CPU activity for the dataloader manager process and its children.

    The monitor owns the dataloader handle and psutil processes internally, so callers
    simply construct it with the dataloader and then call `compute()` whenever metrics
    are needed.
    """

    def __init__(
        self,
        dataloader: object,
        process_factory: Callable[[int], psutil.Process] | None = None,
    ) -> None:
        self.dataloader = dataloader
        self.process_factory = process_factory or psutil.Process
        self._reset_state()

    def attach(self, dataloader: object) -> None:
        if dataloader is self.dataloader:
            return
        self.dataloader = dataloader
        self._reset_state()

    def compute(self, now: float | None = None) -> dict[str, float]:
        if now is None:
            now = time.time()

        metrics: dict[str, float] = {
            "loader/buffer_fill": self._get_reservoir_fill(self.dataloader)
        }

        manager_pid = self._get_manager_pid(self.dataloader)
        if manager_pid <= 0:
            self._reset_state(preserve_warnings=True)
            return metrics

        if self.current_pid != manager_pid:
            self._reset_state()
            self.current_pid = manager_pid

        process = self._ensure_process(manager_pid)
        if process is None:
            return metrics

        self._update_children(process)

        if self.can_read_io:
            read = self._read_bytes(process, now)
            if read is not None:
                read_mb, read_mb_s = read
                metrics["loader/read_mb"] = read_mb
                metrics["loader/read_mb_s"] = read_mb_s

        if self.can_read_cpu:
            cpu_total = 0.0
            for child in self.children:
                cpu = self._read_cpu_percent(child, is_parent=False)
                if cpu is not None:
                    cpu_total += cpu
            parent_cpu = self._read_cpu_percent(process, is_parent=True)
            if parent_cpu is not None:
                cpu_total += parent_cpu
                metrics["loader/cpu_util"] = cpu_total
        else:
            self.warned_cpu = True

        return metrics

    # Internal helpers -----------------------------------------------------------------

    def _reset_state(self, *, preserve_warnings: bool = False) -> None:
        self.last_rb: int | None = None
        self.last_t: float | None = None
        self.current_pid: int | None = None
        self.process: object | None = None
        self.children: list[object] = []
        self.can_read_io = True
        self.can_read_cpu = True
        if not preserve_warnings:
            self.warned_io = False
            self.warned_cpu = False

    def _ensure_process(self, pid: int) -> object | None:
        process = self.process
        if (
            process is None
            or getattr(process, "pid", None) != pid
            or not self._is_running(process)
        ):
            try:
                process = self.process_factory(pid)
            except Exception:  # noqa: BLE001
                return None
            self.process = process
        return process

    @staticmethod
    def _is_running(process: object) -> bool:
        if not hasattr(process, "is_running"):
            return True
        try:
            return bool(process.is_running())
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False
        except Exception:  # noqa: BLE001
            return False

    def _update_children(self, process: object) -> None:
        if not hasattr(process, "children"):
            self.children = []
            return
        try:
            children = process.children(recursive=True)
        except psutil.Error:
            self.children = []
        except Exception:  # noqa: BLE001
            self.children = []
        else:
            self.children = list(children) if children is not None else []

    def _read_bytes(self, process: object, now: float) -> tuple[float, float] | None:
        if not hasattr(process, "io_counters"):
            return None
        try:
            counters = process.io_counters()
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
        ) as err:
            self._disable_io(err)
            return None
        except Exception as err:  # noqa: BLE001
            self._disable_io(err)
            return None

        rb = getattr(counters, "read_bytes", None)
        if rb is None:
            return None

        if self.last_rb is None or self.last_t is None:
            read_mb = 0.0
            read_mb_s = 0.0
        else:
            read_mb = max(rb - self.last_rb, 0) / (1024 * 1024)
            interval = max(now - self.last_t, 1e-6)
            read_mb_s = read_mb / interval
        self.last_rb, self.last_t = rb, now
        return read_mb, read_mb_s

    def _disable_io(self, err: Exception) -> None:
        self.can_read_io = False
        self.last_rb = None
        self.last_t = None
        if not self.warned_io:
            logger.warning("Disabling dataloader IO metrics: %s", err)
            self.warned_io = True

    def _read_cpu_percent(self, process: object, *, is_parent: bool) -> float | None:
        if not hasattr(process, "cpu_percent"):
            return 0.0
        try:
            value = process.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            if is_parent:
                self.can_read_cpu = False
            return None
        except psutil.AccessDenied as err:
            if is_parent:
                self.can_read_cpu = False
                if not self.warned_cpu:
                    logger.warning("Disabling dataloader CPU metrics: %s", err)
                    self.warned_cpu = True
            return None
        except Exception:
            if is_parent:
                self.can_read_cpu = False
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _get_manager_pid(dataloader: object) -> int:
        pid = getattr(dataloader, "manager_pid", None)
        if callable(pid):
            try:
                pid = pid()
            except Exception:  # noqa: BLE001
                return -1
        if pid is None:
            return -1
        try:
            return int(pid)
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _get_reservoir_fill(dataloader: object) -> float:
        reservoir = getattr(dataloader, "reservoir", None)
        if reservoir is None or not hasattr(reservoir, "fill"):
            return 0.0
        try:
            return float(reservoir.fill())
        except Exception:  # noqa: BLE001
            return 0.0
