import beartype
import wandb

MetricQueue = list[tuple[int, dict[str, object]]]


@beartype.beartype
class ParallelWandbRun:
    """
    Inspired by https://community.wandb.ai/t/is-it-possible-to-log-to-multiple-runs-simultaneously/4387/3.
    """

    def __init__(
        self,
        project: str,
        cfgs: list[dict[str, object]],
        mode: str,
        tags: list[str],
        dir: str = ".wandb",
    ):
        cfg, *cfgs = cfgs
        self.project = project
        self.cfgs = cfgs
        self.mode = mode
        self.tags = tags
        self.dir = dir

        self.live_run = wandb.init(
            project=project, config=cfg, mode=mode, tags=tags, dir=dir
        )

        self.metric_queues: list[MetricQueue] = [[] for _ in self.cfgs]

    def log(self, metrics: list[dict[str, object]], *, step: int):
        metric, *metrics = metrics
        self.live_run.log(metric, step=step)
        for queue, metric in zip(self.metric_queues, metrics):
            queue.append((step, metric))

    def finish(self) -> list[str]:
        ids = [self.live_run.id]
        # Log the rest of the runs.
        self.live_run.finish()

        for queue, cfg in zip(self.metric_queues, self.cfgs):
            run = wandb.init(
                project=self.project,
                config=cfg,
                mode=self.mode,
                tags=self.tags + ["queued"],
                dir=self.dir,
            )
            for step, metric in queue:
                run.log(metric, step=step)
            ids.append(run.id)
            run.finish()

        return ids
