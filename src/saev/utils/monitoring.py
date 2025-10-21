import dataclasses
import logging

import beartype
import psutil

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(slots=True)
class LoaderMonitor:
    last_rb: int | None = None
    last_t: float | None = None
    current_pid: int | None = None
    can_read_io: bool = True
    can_read_cpu: bool = True
    warned_io: bool = False
    warned_cpu: bool = False

    def collect(
        self,
        p_dataloader: psutil.Process | None,
        p_children: list[psutil.Process],
        reservoir_fill: float,
        now: float,
    ) -> dict[str, float]:
        if p_dataloader is None:
            self.current_pid = None
            return {}

        if self.current_pid != p_dataloader.pid:
            self.current_pid = p_dataloader.pid
            self.last_rb = None
            self.last_t = None
            self.can_read_io = True
            self.can_read_cpu = True
            self.warned_io = False
            self.warned_cpu = False

        metrics = {"loader/buffer_fill": reservoir_fill}

        if self.can_read_io:
            try:
                io_counters = p_dataloader.io_counters()
            except (
                psutil.AccessDenied,
                psutil.NoSuchProcess,
                psutil.ZombieProcess,
            ) as err:
                self.can_read_io = False
                self.last_rb = None
                self.last_t = None
                if not self.warned_io:
                    logger.warning("Disabling dataloader IO metrics: %s", err)
                    self.warned_io = True
            else:
                rb = io_counters.read_bytes
                if self.last_rb is None or self.last_t is None:
                    read_mb = 0.0
                    read_mb_s = 0.0
                else:
                    read_mb = max(rb - self.last_rb, 0) / (1024 * 1024)
                    interval = max(now - self.last_t, 1e-6)
                    read_mb_s = read_mb / interval
                self.last_rb, self.last_t = rb, now
                metrics["loader/read_mb"] = read_mb
                metrics["loader/read_mb_s"] = read_mb_s

        if self.can_read_cpu:
            cpu_util = 0.0
            for child in p_children:
                try:
                    cpu_util += child.cpu_percent(None)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                except psutil.AccessDenied:
                    continue
            try:
                cpu_util += p_dataloader.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                self.can_read_cpu = False
            except psutil.AccessDenied as err:
                self.can_read_cpu = False
                if not self.warned_cpu:
                    logger.warning("Disabling dataloader CPU metrics: %s", err)
                    self.warned_cpu = True
            else:
                metrics["loader/cpu_util"] = cpu_util

        if not self.can_read_cpu:
            self.warned_cpu = True

        return metrics
