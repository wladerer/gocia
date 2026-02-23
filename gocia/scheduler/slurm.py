"""
gocia/scheduler/slurm.py

Slurm scheduler backend.

Uses sbatch for submission and sacct (preferred) / squeue for status polling.
sacct is preferred because it can report status for completed jobs that have
already left the squeue.  Falls back to squeue if sacct is unavailable.

PBS-to-Slurm differences handled by SlurmRenderer overrides:
  - walltime:       --time
  - tasks_per_node: --ntasks-per-node
  - mem:            --mem (per node)
  - gpus:           --gres=gpu:N
  - account:        --account
  - partition:      --partition
  - qos:            --qos
  - constraint:     --constraint

All of these are the Slurm defaults in ResourceRenderer, so SlurmRenderer
needs no overrides — it exists as a named class for clarity and future
extensibility.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from gocia.scheduler.base import (
    JobStatus,
    ResourceRenderer,
    Scheduler,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slurm resource renderer
# ---------------------------------------------------------------------------

class SlurmRenderer(ResourceRenderer):
    """
    Renders SchedulerResources into ``#SBATCH`` directive lines.

    All methods inherit the correct Slurm syntax from ResourceRenderer.
    This class exists as a named type for clarity.
    """

    @property
    def prefix(self) -> str:
        return "#SBATCH"


# ---------------------------------------------------------------------------
# Status parsing
# ---------------------------------------------------------------------------

# sacct state codes → JobStatus
_SACCT_STATE_MAP: dict[str, JobStatus] = {
    "PENDING":    JobStatus.PENDING,
    "RUNNING":    JobStatus.RUNNING,
    "COMPLETED":  JobStatus.DONE,
    "FAILED":     JobStatus.FAILED,
    "TIMEOUT":    JobStatus.FAILED,
    "CANCELLED":  JobStatus.CANCELLED,
    "CANCELLED+": JobStatus.CANCELLED,
    "NODE_FAIL":  JobStatus.FAILED,
    "OUT_OF_ME":  JobStatus.FAILED,    # OUT_OF_MEMORY, truncated by sacct
    "PREEMPTED":  JobStatus.PENDING,   # will be re-queued
    "SUSPENDED":  JobStatus.PENDING,
    "REQUEUED":   JobStatus.PENDING,
    "COMPLETING":  JobStatus.RUNNING,
}

# squeue state codes → JobStatus
_SQUEUE_STATE_MAP: dict[str, JobStatus] = {
    "PD": JobStatus.PENDING,
    "R":  JobStatus.RUNNING,
    "CG": JobStatus.RUNNING,   # completing
    "CD": JobStatus.DONE,
    "F":  JobStatus.FAILED,
    "TO": JobStatus.FAILED,
    "CA": JobStatus.CANCELLED,
    "NF": JobStatus.FAILED,
    "PR": JobStatus.PENDING,   # preempted
    "S":  JobStatus.PENDING,   # suspended
    "ST": JobStatus.PENDING,   # stopped
}


def _parse_sacct(output: str) -> dict[str, JobStatus]:
    """
    Parse `sacct -j <ids> --format=JobID,State --noheader --parsable2` output.

    Each line is `<jobid>|<state>`.  Array job steps (123_0, 123.batch) are
    collapsed to their parent job ID.
    """
    result: dict[str, JobStatus] = {}
    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue

        raw_id, raw_state = parts[0].strip(), parts[1].strip().upper()

        # Normalise array/step IDs to base job ID
        base_id = re.split(r"[_.]", raw_id)[0]

        # State may have trailing reason in parens: "CANCELLED by 1234"
        state_word = raw_state.split()[0].rstrip("+")

        status = _SACCT_STATE_MAP.get(state_word, JobStatus.UNKNOWN)

        # Upgrade to worst status seen for this job (FAILED > RUNNING > PENDING)
        _PRIORITY = {
            JobStatus.FAILED: 5,
            JobStatus.CANCELLED: 4,
            JobStatus.UNKNOWN: 3,
            JobStatus.DONE: 2,
            JobStatus.RUNNING: 1,
            JobStatus.PENDING: 0,
        }
        existing = result.get(base_id)
        if existing is None or _PRIORITY.get(status, 0) > _PRIORITY.get(existing, 0):
            result[base_id] = status

    return result


def _parse_squeue(output: str) -> dict[str, JobStatus]:
    """
    Parse `squeue -j <ids> --format=%i %t --noheader` output.

    Each line: `<jobid> <state_code>`.
    """
    result: dict[str, JobStatus] = {}
    for line in output.strip().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        job_id, state_code = parts[0], parts[1].upper()
        result[job_id] = _SQUEUE_STATE_MAP.get(state_code, JobStatus.UNKNOWN)
    return result


# ---------------------------------------------------------------------------
# Slurm scheduler
# ---------------------------------------------------------------------------

class SlurmScheduler(Scheduler):
    """
    Slurm scheduler backend.

    Submits via ``sbatch``, polls via ``sacct`` (with ``squeue`` fallback),
    cancels via ``scancel``.
    """

    renderer = SlurmRenderer()

    def submit(self, script_path: Path) -> str:
        """
        Submit a job script via sbatch.

        Returns
        -------
        str
            Numeric Slurm job ID.

        Raises
        ------
        RuntimeError
            If sbatch fails or returns a non-parseable job ID.
        """
        result = self._run(["sbatch", str(script_path)])
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (exit {result.returncode}):\n{result.stderr.strip()}"
            )

        # sbatch stdout: "Submitted batch job 123456"
        match = re.search(r"(\d+)", result.stdout)
        if not match:
            raise RuntimeError(
                f"Could not parse job ID from sbatch output: {result.stdout!r}"
            )

        job_id = match.group(1)
        logger.info(f"  Submitted Slurm job {job_id}: {script_path.name}")
        return job_id

    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """
        Query job statuses via sacct, falling back to squeue.

        Jobs not found in either command are reported as DONE (they likely
        completed and aged out of the scheduler's memory).
        """
        if not job_ids:
            return {}

        ids_str = ",".join(job_ids)

        # Try sacct first (can see completed jobs)
        result = self._run([
            "sacct",
            "-j", ids_str,
            "--format=JobID,State",
            "--noheader",
            "--parsable2",
        ])

        if result.returncode == 0 and result.stdout.strip():
            statuses = _parse_sacct(result.stdout)
        else:
            # Fallback to squeue (only sees queued/running jobs)
            logger.debug("sacct unavailable or returned empty; falling back to squeue")
            result = self._run([
                "squeue",
                "-j", ids_str,
                "--format=%i %t",
                "--noheader",
            ])
            statuses = _parse_squeue(result.stdout) if result.returncode == 0 else {}

        # Jobs not appearing in output are assumed done
        for job_id in job_ids:
            if job_id not in statuses:
                statuses[job_id] = JobStatus.DONE

        return statuses

    def cancel(self, job_id: str) -> None:
        """Cancel a job via scancel."""
        result = self._run(["scancel", job_id])
        if result.returncode != 0:
            logger.warning(f"scancel {job_id} returned {result.returncode}: {result.stderr.strip()}")
        else:
            logger.info(f"  Cancelled Slurm job {job_id}")
