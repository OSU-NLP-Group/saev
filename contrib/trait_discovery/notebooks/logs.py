import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import dataclasses
    import datetime
    import json
    import pathlib
    import typing as tp

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt

    import saev.helpers

    return beartype, dataclasses, datetime, json, mo, pathlib, plt, saev, tp


@app.cell
def _(mo):
    mo.md(
        """
    # Logs

    This notebook parses logs from a probe1d.py run and explores VRAM usage, loss and gradients.
    """
    )
    return


@app.cell
def _(mo):
    log_input = mo.ui.text(label="Log Path:", full_width=True)
    return (log_input,)


@app.cell
def _(log_input):
    log_input
    return


@app.cell
def _(load_events, log_input, mo, pathlib):
    log_fpath = pathlib.Path(log_input.value)
    mo.stop(not log_fpath.is_file())
    events = load_events(log_fpath)
    len(events)
    return (events,)


@app.cell
def _(beartype, dataclasses, datetime, json, pathlib, tp):
    @beartype.beartype
    def _to_float(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"invalid float value: {value}")

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Event:
        timestamp: datetime.datetime
        name: str

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class ProbeIter(Event):
        # {"timestamp": "2025-11-03T22:14:15.545037+00:00", "metric": "probe_iteration", "slab": [80, 88], "iter": 27, "grad_max": 0.00024337830836884677, "step_max": 0.06957802176475525, "lambda_mean": 982822682624.0, "loss_mean": 0.005730595905333757, "loss_max": 0.006768315099179745, "rho_mean": 400409.9375, "rho_min": -20489095168.0, "success_frac": 1.0, "fallback": 0, "step_clipped": 0, "pred_mean": 5.661942789614294e-11, "peak_alloc_mb": 11503.52734375, "peak_reserved_mb": 11578.0}
        slab: tuple[int, int]
        iter: int
        grad_max: float | None
        step_max: float | None
        lambda_mean: float | None
        loss_mean: float | None
        loss_max: float | None
        rho_mean: float | None
        rho_min: float | None
        success_frac: float | None
        fallback: int
        step_clipped: int
        pred_mean: float | None
        peak_alloc_gb: float | None
        peak_reserved_gb: float | None
        rss_gb: float | None

        @classmethod
        def from_payload(cls, payload: dict[str, object]) -> tp.Self:
            try:
                timestamp = datetime.datetime.fromisoformat(payload["timestamp"])
                slab_raw = payload["slab"]
                iter_idx = int(payload["iter"])
                fallback = int(payload["fallback"])
                step_clipped = int(payload["step_clipped"])
            except (KeyError, ValueError, TypeError) as err:
                raise ValueError("invalid probe iteration payload") from err

            if not isinstance(slab_raw, (list, tuple)) or len(slab_raw) != 2:
                raise ValueError("slab must be length-2 sequence")
            slab = (int(slab_raw[0]), int(slab_raw[1]))

            kwargs = {
                "timestamp": timestamp,
                "name": "probe_iteration",
                "slab": slab,
                "iter": iter_idx,
                "grad_max": _to_float(payload.get("grad_max")),
                "step_max": _to_float(payload.get("step_max")),
                "lambda_mean": _to_float(payload.get("lambda_mean")),
                "loss_mean": _to_float(payload.get("loss_mean")),
                "loss_max": _to_float(payload.get("loss_max")),
                "rho_mean": _to_float(payload.get("rho_mean")),
                "rho_min": _to_float(payload.get("rho_min")),
                "success_frac": _to_float(payload.get("success_frac")),
                "fallback": fallback,
                "step_clipped": step_clipped,
                "pred_mean": _to_float(payload.get("pred_mean")),
                "peak_alloc_gb": _to_float(payload.get("peak_alloc_mb")) / 1024,
                "peak_reserved_gb": _to_float(payload.get("peak_reserved_mb")) / 1024,
                "rss_gb": _to_float(payload.get("rss_gb")),
            }
            return cls(**kwargs)

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class LoadCsrRamStart(Event):
        split: tp.Literal["train", "test"]
        fpath: str
        rss_gb: float

        @classmethod
        def from_payload(cls, payload: dict[str, object]) -> tp.Self:
            try:
                timestamp = datetime.datetime.fromisoformat(payload["timestamp"])
            except (KeyError, ValueError, TypeError) as err:
                raise ValueError("invalid probe iteration payload") from err

            return cls(
                timestamp=timestamp,
                name="load_csr_ram_start",
                split=payload["split"],
                fpath=payload["fpath"],
                rss_gb=payload["rss_gb"],
            )

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class LoadCsrRamEnd(Event):
        rss_gb: float
        split: tp.Literal["train", "test"]
        nnz: int
        shape: tuple[int, ...]

        @classmethod
        def from_payload(cls, payload: dict[str, object]) -> tp.Self:
            try:
                timestamp = datetime.datetime.fromisoformat(payload["timestamp"])
            except (KeyError, ValueError, TypeError) as err:
                raise ValueError("invalid probe iteration payload") from err

            return cls(
                timestamp=timestamp,
                name="load_csr_ram_end",
                rss_gb=payload["rss_gb"],
                split=payload["split"],
                nnz=payload["nnz"],
                shape=tuple(payload["shape"]),
            )

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class LoadCsrVramStart(Event):
        rss_gb: float
        split: tp.Literal["train", "test"]
        device: str

        @classmethod
        def from_payload(cls, payload: dict[str, object]) -> tp.Self:
            try:
                timestamp = datetime.datetime.fromisoformat(payload["timestamp"])
            except (KeyError, ValueError, TypeError) as err:
                raise ValueError("invalid probe iteration payload") from err

            return cls(
                timestamp=timestamp,
                name="load_csr_vram_start",
                rss_gb=payload["rss_gb"],
                split=payload["split"],
                device=payload["device"],
            )

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class LoadCsrVramEnd(Event):
        rss_gb: float
        split: tp.Literal["train", "test"]
        device: str

        @classmethod
        def from_payload(cls, payload: dict[str, object]) -> tp.Self:
            try:
                timestamp = datetime.datetime.fromisoformat(payload["timestamp"])
            except (KeyError, ValueError, TypeError) as err:
                raise ValueError("invalid probe iteration payload") from err

            return cls(
                timestamp=timestamp,
                name="load_csr_vram_end",
                rss_gb=payload["rss_gb"],
                split=payload["split"],
                device=payload["device"],
            )

    @beartype.beartype
    def load_event(line: str) -> Event:
        start = line.find("{")
        if start < 0:
            raise ValueError(f"missing json payload in '{line.strip()}'")
        payload = json.loads(line[start:])

        if not isinstance(payload, dict):
            raise ValueError("payload must be object")

        event = None
        for key in ("event", "metric"):
            if key in payload:
                event = payload[key]
                break

        if event == "probe_iteration":
            return ProbeIter.from_payload(payload)
        elif event == "load_csr_ram_start":
            return LoadCsrRamStart.from_payload(payload)
        elif event == "load_csr_ram_end":
            return LoadCsrRamEnd.from_payload(payload)
        elif event == "load_csr_vram_start":
            return LoadCsrVramStart.from_payload(payload)
        elif event == "load_csr_vram_end":
            return LoadCsrVramEnd.from_payload(payload)
        else:
            raise ValueError(f"Unexpected event '{event}'")

    @beartype.beartype
    def load_events(log_fpath: pathlib.Path) -> list[Event]:
        iters: list[ProbeIter] = []
        with log_fpath.open("r", encoding="utf-8") as fd:
            for line in fd:
                if not line.strip():
                    continue

                if "[stats]" not in line:
                    continue

                try:
                    iter_item = load_event(line)
                except ValueError as err:
                    print(err)
                    continue

                iters.append(iter_item)
        return iters

    return ProbeIter, load_events


@app.cell
def _(events, plt, saev):
    def _():
        slabs = saev.helpers.batched_idx(151, 8)
        fig, ax1 = plt.subplots(dpi=200, figsize=(6, 3), layout="constrained")

        # Set to None for all events.
        # max_events = None
        max_events = 50

        def attrs(attr: str) -> list:
            return [
                getattr(e, attr) if hasattr(e, attr) else 0 for e in events[:max_events]
            ]

        xs = range(1, len(events[:max_events]) + 1)

        ax1.plot(
            xs,
            attrs("peak_alloc_gb"),
            color="tab:blue",
            label="Peak Alloc. VRAM",
            alpha=0.5,
            marker=".",
        )
        ax1.plot(
            xs,
            attrs("peak_reserved_gb"),
            color="tab:orange",
            label="Peak Res. VRAM",
            alpha=0.5,
            marker=".",
        )
        ax1.set_ylabel("VRAM GB")
        # ax1.set_xscale("log")

        ax2 = ax1.twinx()
        ax2.plot(
            xs,
            attrs("rss_gb"),
            color="tab:green",
            label="RAM",
            alpha=0.5,
            marker=".",
        )
        ax2.set_ylabel("RAM GB")

        ax1.legend(loc="lower right")
        ax2.legend()

        return fig

    _()
    return


@app.cell
def _(ProbeIter, events, mo, plt, saev):
    def _():
        slabs = saev.helpers.batched_idx(151, 8)
        fig, axes = plt.subplots(
            nrows=len(slabs),
            ncols=2,
            dpi=200,
            sharex=True,
            figsize=(12, 4 * len(slabs)),
            layout="constrained",
        )

        for slab, (ax1, ax2) in zip(mo.status.progress_bar(slabs), axes):
            slab_iters = [
                it for it in events if isinstance(it, ProbeIter) and it.slab == slab
            ]

            def attrs(attr: str) -> list:
                return [getattr(it, attr) for it in slab_iters]

            xs = [it.iter for it in slab_iters]

            ax1.spines[["right", "top"]].set_visible(False)
            ax1.set_yscale("log")
            ax1.set_title(f"Metrics for Slab {slab}")

            ax2.spines[["right", "top"]].set_visible(False)
            ax2.set_title(f"Optim. for Slab {slab}")

            if not xs:
                continue

            ax1.plot(xs, attrs("grad_max"), color="tab:blue", label="Max Grad.")
            ax1.plot(xs, attrs("step_max"), color="tab:orange", label="Max Step")
            ax1.plot(xs, attrs("loss_mean"), color="tab:green", label="Mean Loss")
            ax1.plot(xs, attrs("loss_max"), color="tab:purple", label="Max Loss")
            ax1.plot(
                xs, attrs("success_frac"), color="tab:brown", label="Success Frac."
            )

            # n_nonzero_step_clipped = sum(it.step_clipped > 0 for it in slab_iters)

            # if 0 < n_nonzero_step_clipped < 5:
            #     ax1.scatter(
            #         xs, attrs("step_clipped"), color="tab:red", label="Clipped Steps"
            #     )

            # if 5 <= n_nonzero_step_clipped:
            #     ax1.plot(xs, attrs("step_clipped"), color="tab:red", label="Clipped Steps")

            # AX 3
            ax2.plot(xs, attrs("rho_mean"), color="tab:blue", label="Mean $\\rho$")
            ax2.plot(xs, attrs("rho_min"), color="tab:orange", label="Min $\\rho$")
            ax2.plot(
                xs, attrs("lambda_mean"), color="tab:green", label="Mean $\\lambda$"
            )
            ax2.set_yscale("symlog")

            # Legends
            ax1.legend()
            ax2.legend()

        return fig

    _()
    return


if __name__ == "__main__":
    app.run()
