"""
gocia/scheduler/local.py

Local scheduler: runs jobs as subprocesses in the current environment.

Intended for development and testing without an HPC cluster.  Jobs are
run synchronously (blocking) up to nworkers at a time using a thread pool.

The local scheduler does not write job scripts — it runs the body directly
via subprocess.  The ``walltime`` and ``resources`` fields in SchedulerConfig
are ignored; this is intentional since resource limits are not enforceable
in a local subprocess context.

submit() returns an integer string job ID that is just a counter.  status()
always returns DONE for completed jobs and RUNNING for active ones.

Usage
-----
    scheduler:
      type: local
      nworkers: 2
      walltime: "00:10:00"   # ignored for local
"""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

from gocia.scheduler.base import JobStatus, ResourceRenderer, Scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Null renderer (local jobs don't need scheduler directives)
# ---------------------------------------------------------------------------

class _NullRenderer(ResourceRenderer):
    """Renderer that produces no output — local jobs don't use job scripts."""

    @property
    def prefix(self) -> str:
        return "#LOCAL"

    def render_resources(self, resources, walltime: str) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Local scheduler
# ---------------------------------------------------------------------------

class LocalScheduler(Scheduler):
    """
    Local subprocess scheduler for testing without an HPC cluster.

    Jobs are run directly as subprocesses.  Concurrency is limited to
    config.nworkers using a threading.Semaphore.

    Because jobs run synchronously within submit(), the status() method
    will always see them as DONE once submit() returns.  This is correct
    for the local case: the runner loop calls submit() and immediately moves
    on, relying on sentinel files (written by the pipeline) for state.
    """

    renderer = _NullRenderer()

    def __init__(self, config) -> None:
        super().__init__(config)
        self._semaphore = threading.Semaphore(config.nworkers)
        self._counter = 0
        self._counter_lock = threading.Lock()
        # job_id → (status, thread)
        self._jobs: dict[str, dict] = {}

    def _next_id(self) -> str:
        with self._counter_lock:
            self._counter += 1
            return str(self._counter)

    def submit(self, script_path: Path) -> str:
        """
        Run a job script as a subprocess.

        Acquires a semaphore slot (blocks if nworkers jobs are already
        running), then runs the script in a background thread so the
        calling thread is not blocked waiting for the job to finish.

        Parameters
        ----------
        script_path:
            Path to a .sh script.  Must be executable.

        Returns
        -------
        str
            A simple integer job ID string (local counter).
        """
        job_id = self._next_id()
        self._jobs[job_id] = {"status": JobStatus.PENDING, "thread": None}

        def _run_job():
            self._semaphore.acquire()
            try:
                self._jobs[job_id]["status"] = JobStatus.RUNNING
                logger.info(f"  [local] Job {job_id} started: {script_path.name}")

                result = subprocess.run(
                    ["bash", str(script_path)],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    self._jobs[job_id]["status"] = JobStatus.DONE
                    logger.info(f"  [local] Job {job_id} completed OK")
                else:
                    self._jobs[job_id]["status"] = JobStatus.FAILED
                    logger.error(
                        f"  [local] Job {job_id} FAILED (exit {result.returncode}):\n"
                        f"{result.stderr[-500:]}"   # last 500 chars of stderr
                    )
            finally:
                self._semaphore.release()

        thread = threading.Thread(target=_run_job, daemon=True)
        thread.start()
        self._jobs[job_id]["thread"] = thread

        return job_id

    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """
        Return current status for each job ID.

        Jobs not in the local registry are returned as DONE (they must
        have been submitted in a previous process invocation).
        """
        result = {}
        for job_id in job_ids:
            if job_id in self._jobs:
                result[job_id] = self._jobs[job_id]["status"]
            else:
                result[job_id] = JobStatus.DONE
        return result

    def cancel(self, job_id: str) -> None:
        """
        Cancel a local job.

        Because threads are not easily cancellable in Python, this marks
        the job as CANCELLED in the registry but does not actually kill
        the subprocess.  For local testing this is acceptable behaviour.
        """
        if job_id in self._jobs:
            logger.warning(
                f"[local] Cannot forcibly kill job {job_id} "
                "(Python threads are not cancellable). Marked as CANCELLED."
            )
            self._jobs[job_id]["status"] = JobStatus.CANCELLED

    def wait_all(self, timeout: float | None = None) -> None:
        """
        Block until all submitted jobs have finished.

        Useful in tests to ensure all pipelines complete before asserting
        on results.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait per thread (None = wait indefinitely).
        """
        for job_id, job in self._jobs.items():
            thread = job.get("thread")
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout)

    def submit_structure(
        self,
        job_name: str,
        body: str,
        work_dir: Path,
    ) -> str:
        """
        Write body to a script file and submit it.

        Overrides the base class to skip the scheduler-directive header —
        local scripts are plain shell scripts with just the body.
        """
        script_path = work_dir / f"{job_name}.sh"
        script_path.write_text(f"#!/bin/bash\nset -e\n\n{body}\n")
        script_path.chmod(0o755)
        return self.submit(script_path)
