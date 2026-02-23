"""
gocia/scheduler/base.py

Abstract Scheduler base class and shared job script rendering.

Every scheduler (Slurm, PBS, local) shares one job script template.
Rendering is handled here; scheduler-specific directive formatting is
delegated to each backend via render_resources().

Job script structure
--------------------
    #!/bin/bash
    #<prefix> --job-name=<name>
    #<prefix> --output=<logfile>
    #<prefix> --error=<errfile>
    #<prefix> --time=<walltime>
    ... structured resource directives ...
    ... extra_directives verbatim ...
    <blank line>
    <body>

where <prefix> is "#SBATCH" for Slurm, "#PBS" for PBS.

Public API
----------
    render_job_script(scheduler, job_name, body, work_dir) -> str
    parse_job_id(output) -> str

    # Abstract interface every backend must implement:
    submit(script_path) -> job_id
    status(job_ids) -> dict[job_id, JobStatus]
    cancel(job_id) -> None
"""

from __future__ import annotations

import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Job status enum
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    """Normalised job state shared across all scheduler backends."""
    PENDING   = "pending"    # queued, not yet running
    RUNNING   = "running"    # actively running on nodes
    DONE      = "done"       # completed successfully
    FAILED    = "failed"     # exited with non-zero status
    CANCELLED = "cancelled"  # user or admin cancelled
    UNKNOWN   = "unknown"    # scheduler returned an unrecognised state


# ---------------------------------------------------------------------------
# Resource renderer — produces per-scheduler directive lines
# ---------------------------------------------------------------------------

class ResourceRenderer(ABC):
    """
    Renders a SchedulerResources model into scheduler-specific directive lines.

    Each backend subclasses this and overrides individual methods for any
    directives that differ from the Slurm convention.
    """

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Directive prefix, e.g. '#SBATCH' or '#PBS'."""
        ...

    def _line(self, option: str, value: Any = None) -> str:
        """Format a single directive line."""
        if value is None:
            return f"{self.prefix} {option}"
        return f"{self.prefix} {option}={value}"

    # ------------------------------------------------------------------
    # Individual resource renderers — override in subclasses as needed
    # ------------------------------------------------------------------

    def walltime(self, walltime: str) -> list[str]:
        return [self._line("--time", walltime)]

    def job_name(self, name: str) -> list[str]:
        return [self._line("--job-name", name)]

    def output(self, path: str) -> list[str]:
        return [self._line("--output", path)]

    def error(self, path: str) -> list[str]:
        return [self._line("--error", path)]

    def nodes(self, n: int) -> list[str]:
        return [self._line("--nodes", n)]

    def tasks_per_node(self, n: int) -> list[str]:
        return [self._line("--ntasks-per-node", n)]

    def cpus_per_task(self, n: int) -> list[str]:
        return [self._line("--cpus-per-task", n)]

    def mem(self, value: str) -> list[str]:
        return [self._line("--mem", value)]

    def mem_per_cpu(self, value: str) -> list[str]:
        return [self._line("--mem-per-cpu", value)]

    def gpus(self, n: int) -> list[str]:
        return [self._line("--gres", f"gpu:{n}")]

    def account(self, value: str) -> list[str]:
        return [self._line("--account", value)]

    def partition(self, value: str) -> list[str]:
        return [self._line("--partition", value)]

    def qos(self, value: str) -> list[str]:
        return [self._line("--qos", value)]

    def constraint(self, value: str) -> list[str]:
        return [self._line("--constraint", value)]

    # ------------------------------------------------------------------
    # Top-level renderer
    # ------------------------------------------------------------------

    def render_resources(self, resources, walltime: str) -> list[str]:
        """
        Render all non-None resource fields into directive lines.

        Parameters
        ----------
        resources:
            SchedulerResources model instance.
        walltime:
            Walltime string from SchedulerConfig.

        Returns
        -------
        list[str]
            One directive line per resource field that is set.
        """
        lines: list[str] = []
        lines += self.walltime(walltime)

        if resources.nodes is not None:
            lines += self.nodes(resources.nodes)
        if resources.tasks_per_node is not None:
            lines += self.tasks_per_node(resources.tasks_per_node)
        if resources.cpus_per_task is not None:
            lines += self.cpus_per_task(resources.cpus_per_task)
        if resources.mem is not None:
            lines += self.mem(resources.mem)
        if resources.mem_per_cpu is not None:
            lines += self.mem_per_cpu(resources.mem_per_cpu)
        if resources.gpus is not None:
            lines += self.gpus(resources.gpus)
        if resources.account is not None:
            lines += self.account(resources.account)
        if resources.partition is not None:
            lines += self.partition(resources.partition)
        if resources.qos is not None:
            lines += self.qos(resources.qos)
        if resources.constraint is not None:
            lines += self.constraint(resources.constraint)

        return lines


# ---------------------------------------------------------------------------
# Job script builder
# ---------------------------------------------------------------------------

def render_job_script(
    renderer: ResourceRenderer,
    config,             # SchedulerConfig
    job_name: str,
    body: str,
    work_dir: Path,
) -> str:
    """
    Render a complete job script as a string.

    Parameters
    ----------
    renderer:
        A ResourceRenderer for the target scheduler (Slurm/PBS).
    config:
        The SchedulerConfig instance.
    job_name:
        Job name (used in scheduler dashboard and log filenames).
    body:
        Shell commands to execute inside the job (after the header).
    work_dir:
        Working directory where log files will be written.

    Returns
    -------
    str
        Complete job script ready to write to a .sh file.
    """
    lines = ["#!/bin/bash"]

    # Standard identification/log directives
    lines += renderer.job_name(job_name)
    lines += renderer.output(str(work_dir / f"{job_name}.out"))
    lines += renderer.error(str(work_dir / f"{job_name}.err"))

    # Structured resource directives
    lines += renderer.render_resources(config.resources, config.walltime)

    # Extra verbatim directives — normalise: strip leading # and scheduler prefix
    for directive in config.extra_directives:
        d = directive.strip()
        # Ensure the line carries the correct prefix
        if d.startswith("#"):
            lines.append(d)
        else:
            lines.append(f"{renderer.prefix} {d}")

    lines.append("")   # blank line before body
    lines.append(body)
    lines.append("")   # trailing newline

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract Scheduler
# ---------------------------------------------------------------------------

class Scheduler(ABC):
    """
    Abstract base class for HPC job schedulers.

    Subclasses implement submit(), status(), and cancel() for a specific
    scheduler (Slurm, PBS) or execution model (local).

    The renderer attribute must be set in each subclass.
    """

    renderer: ResourceRenderer

    def __init__(self, config) -> None:
        self.config = config   # SchedulerConfig

    # ------------------------------------------------------------------
    # Must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def submit(self, script_path: Path) -> str:
        """
        Submit a job script and return the scheduler-assigned job ID.

        Parameters
        ----------
        script_path:
            Path to a .sh file containing the job script.

        Returns
        -------
        str
            Job ID string (numeric for Slurm/PBS).
        """
        ...

    @abstractmethod
    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """
        Query the status of one or more jobs.

        Parameters
        ----------
        job_ids:
            List of job ID strings previously returned by submit().

        Returns
        -------
        dict[str, JobStatus]
            Mapping of job_id → JobStatus.  Jobs not found in the scheduler
            (e.g. too old to appear in queue) are returned as DONE or UNKNOWN.
        """
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Cancel a running or pending job."""
        ...

    # ------------------------------------------------------------------
    # Convenience: build and submit in one call
    # ------------------------------------------------------------------

    def submit_structure(
        self,
        job_name: str,
        body: str,
        work_dir: Path,
    ) -> str:
        """
        Render a job script, write it to work_dir, and submit it.

        Parameters
        ----------
        job_name:
            Short identifier used in the script filename and scheduler name.
        body:
            Shell commands to run (typically: cd work_dir && run_pipeline ...).
        work_dir:
            Directory where the script and log files are written.

        Returns
        -------
        str
            Job ID.
        """
        script = render_job_script(
            renderer=self.renderer,
            config=self.config,
            job_name=job_name,
            body=body,
            work_dir=work_dir,
        )
        script_path = work_dir / f"{job_name}.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)
        return self.submit(script_path)

    # ------------------------------------------------------------------
    # Shared subprocess helper
    # ------------------------------------------------------------------

    @staticmethod
    def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a subprocess and return CompletedProcess."""
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(config) -> "Scheduler":
    """
    Instantiate the appropriate Scheduler subclass from a SchedulerConfig.

    Parameters
    ----------
    config:
        A SchedulerConfig instance.

    Returns
    -------
    Scheduler
    """
    from gocia.scheduler.slurm import SlurmScheduler
    from gocia.scheduler.pbs   import PBSScheduler
    from gocia.scheduler.local import LocalScheduler

    return {
        "slurm": SlurmScheduler,
        "pbs":   PBSScheduler,
        "local": LocalScheduler,
    }[config.type](config)
