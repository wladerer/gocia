"""
gocia/scheduler/pbs.py

PBS/Torque scheduler backend.

PBS uses a different directive syntax from Slurm for most resources.
PBSRenderer overrides each method where PBS syntax differs.

Key differences vs Slurm
-------------------------
walltime        #PBS -l walltime=HH:MM:SS
nodes/tasks     #PBS -l select=N:ncpus=T:mpiprocs=T:ompthreads=C
mem             #PBS -l mem=32gb          (note lowercase 'gb', not 'G')
gpus            #PBS -l ngpus=N
account         #PBS -A account
partition/queue #PBS -q queuename
qos             #PBS -q (PBS conflates queue and QOS — we use -q for both;
                  if both are set we emit a warning and use partition)
constraint      No standard equivalent — emitted as a comment with warning.
output/error    #PBS -o / #PBS -e
job name        #PBS -N

PBS ``select`` statement
------------------------
The resource request for nodes/CPUs in PBS is more complex than Slurm.
Rather than independent --nodes and --ntasks-per-node directives, PBS
wants a single ``select`` statement:

    #PBS -l select=<nodes>:ncpus=<tasks_per_node>:mpiprocs=<tasks_per_node>[:ompthreads=<cpus_per_task>][:ngpus=<gpus>][:mem=<mem>gb]

We build this from the structured resource fields and emit it as one line.
If only some fields are set, we include only those colons.
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
# PBS resource renderer
# ---------------------------------------------------------------------------

class PBSRenderer(ResourceRenderer):
    """
    Renders SchedulerResources into ``#PBS`` directive lines using PBS syntax.
    """

    @property
    def prefix(self) -> str:
        return "#PBS"

    def _line(self, option: str, value=None) -> str:
        """PBS directives use space-separated flag syntax, not '='."""
        if value is None:
            return f"{self.prefix} {option}"
        return f"{self.prefix} {option} {value}"

    # ------------------------------------------------------------------
    # Override: PBS uses -l walltime=
    # ------------------------------------------------------------------

    def walltime(self, walltime: str) -> list[str]:
        return [self._line("-l", f"walltime={walltime}")]

    def job_name(self, name: str) -> list[str]:
        # PBS job names max 15 chars — truncate silently
        return [self._line("-N", name[:15])]

    def output(self, path: str) -> list[str]:
        return [self._line("-o", path)]

    def error(self, path: str) -> list[str]:
        return [self._line("-e", path)]

    def account(self, value: str) -> list[str]:
        return [self._line("-A", value)]

    def partition(self, value: str) -> list[str]:
        return [self._line("-q", value)]

    def qos(self, value: str) -> list[str]:
        # PBS has no separate QOS concept — warn and emit as queue if partition unset
        logger.warning(
            "PBS does not have a native QOS directive. "
            "If 'partition' is also set, 'qos' is ignored. "
            "Otherwise 'qos' is emitted as -q."
        )
        return [self._line("-q", value)]

    def constraint(self, value: str) -> list[str]:
        # No standard PBS equivalent — emit as a comment so the script is still valid
        logger.warning(
            f"PBS has no standard 'constraint' directive. "
            f"Emitting as a comment: # constraint={value}"
        )
        return [f"# constraint={value}  # PBS: no standard equivalent"]

    # ------------------------------------------------------------------
    # Override: build a single PBS 'select' statement for all node/CPU/GPU/mem
    # resources rather than individual directives
    # ------------------------------------------------------------------

    def nodes(self, n: int) -> list[str]:
        return []   # handled by render_resources override

    def tasks_per_node(self, n: int) -> list[str]:
        return []   # handled by render_resources override

    def cpus_per_task(self, n: int) -> list[str]:
        return []   # handled by render_resources override

    def mem(self, value: str) -> list[str]:
        return []   # handled by render_resources override

    def mem_per_cpu(self, value: str) -> list[str]:
        return []   # handled by render_resources override

    def gpus(self, n: int) -> list[str]:
        return []   # handled by render_resources override

    # ------------------------------------------------------------------
    # Top-level override: build select statement + remaining directives
    # ------------------------------------------------------------------

    def render_resources(self, resources, walltime: str) -> list[str]:
        lines: list[str] = []

        # walltime always first
        lines += self.walltime(walltime)

        # Build select statement from node/cpu/gpu/mem fields
        select_parts = _build_select(resources)
        if select_parts:
            lines.append(self._line("-l", f"select={select_parts}"))

        # account, partition, qos, constraint
        # Note: if both partition and qos are set, partition wins and qos is skipped
        if resources.account is not None:
            lines += self.account(resources.account)

        if resources.partition is not None:
            lines.append(self._line("-q", resources.partition))
        elif resources.qos is not None:
            lines += self.qos(resources.qos)

        if resources.constraint is not None:
            lines += self.constraint(resources.constraint)

        return lines


def _build_select(resources) -> str:
    """
    Build the PBS select string from resource fields.

    Returns empty string if no node/cpu/gpu/mem fields are set.

    Example output: "2:ncpus=16:mpiprocs=16:ompthreads=4:ngpus=1:mem=32gb"
    """
    parts: list[str] = []

    n_nodes = resources.nodes or 1
    parts.append(str(n_nodes))

    if resources.tasks_per_node is not None:
        parts.append(f"ncpus={resources.tasks_per_node}")
        parts.append(f"mpiprocs={resources.tasks_per_node}")

    if resources.cpus_per_task is not None:
        parts.append(f"ompthreads={resources.cpus_per_task}")

    if resources.gpus is not None:
        parts.append(f"ngpus={resources.gpus}")

    if resources.mem is not None:
        # Normalise memory unit: PBS expects lowercase 'gb', 'mb', etc.
        mem_pbs = _normalise_mem_pbs(resources.mem)
        parts.append(f"mem={mem_pbs}")

    if resources.mem_per_cpu is not None:
        mem_pbs = _normalise_mem_pbs(resources.mem_per_cpu)
        parts.append(f"mem_per_cpu={mem_pbs}")

    # Only emit select if there's something beyond the bare node count
    if len(parts) == 1 and resources.nodes is None:
        return ""

    return ":".join(parts)


def _normalise_mem_pbs(mem: str) -> str:
    """
    Convert memory string to PBS lowercase format.

    "32G" → "32gb", "128GB" → "128gb", "512M" → "512mb"
    """
    mem = mem.strip()
    match = re.match(r"^(\d+)\s*([GMgm])[Bb]?$", mem)
    if match:
        value, unit = match.group(1), match.group(2).upper()
        return f"{value}{'gb' if unit == 'G' else 'mb'}"
    # Return as-is if we can't parse it — PBS will error if it's wrong
    logger.warning(f"Could not normalise memory string '{mem}' for PBS; using as-is.")
    return mem.lower()


# ---------------------------------------------------------------------------
# Status parsing
# ---------------------------------------------------------------------------

# qstat state codes → JobStatus
_QSTAT_STATE_MAP: dict[str, JobStatus] = {
    "Q": JobStatus.PENDING,     # queued
    "W": JobStatus.PENDING,     # waiting (dependencies)
    "H": JobStatus.PENDING,     # held
    "T": JobStatus.PENDING,     # being transferred
    "R": JobStatus.RUNNING,
    "E": JobStatus.RUNNING,     # exiting (still running cleanup)
    "C": JobStatus.DONE,        # completed
    "F": JobStatus.DONE,        # finished (Torque variant of C)
    "X": JobStatus.DONE,        # expired (PBS Pro)
    "S": JobStatus.PENDING,     # suspended
    "M": JobStatus.PENDING,     # moved to another server
    "B": JobStatus.RUNNING,     # begun (array job has started)
}


def _parse_qstat(output: str) -> dict[str, JobStatus]:
    """
    Parse `qstat -x -f -F json <job_ids>` or plain `qstat <job_ids>` output.

    Tries JSON first (PBS Pro), falls back to tabular format (Torque).
    """
    output = output.strip()
    if not output:
        return {}

    # Try JSON (PBS Pro / OpenPBS)
    if output.startswith("{"):
        try:
            return _parse_qstat_json(output)
        except Exception as exc:
            logger.debug(f"qstat JSON parse failed ({exc}); trying tabular")

    return _parse_qstat_tabular(output)


def _parse_qstat_json(output: str) -> dict[str, JobStatus]:
    """Parse PBS Pro JSON qstat output."""
    import json
    data = json.loads(output)
    jobs = data.get("Jobs", {})
    result = {}
    for job_id, job_data in jobs.items():
        # Normalise job ID: strip server suffix "123.server" → "123"
        base_id = job_id.split(".")[0]
        state = job_data.get("job_state", "?").upper()
        result[base_id] = _QSTAT_STATE_MAP.get(state, JobStatus.UNKNOWN)
    return result


def _parse_qstat_tabular(output: str) -> dict[str, JobStatus]:
    """
    Parse classic Torque qstat tabular output.

    Format (varies by site, best-effort):
        Job id      Name      User      Time  S  Queue
        --------    ------    ------    ----  -  -----
        123.host    myjob     user      0     R  normal
    """
    result = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        raw_id = parts[0]
        # State is typically column 4 (0-indexed) or 5 depending on format
        # Try to find a single-char state in the line
        state_char = None
        for p in parts:
            if len(p) == 1 and p.upper() in _QSTAT_STATE_MAP:
                state_char = p.upper()
                break
        if state_char is None:
            continue
        base_id = raw_id.split(".")[0]
        result[base_id] = _QSTAT_STATE_MAP.get(state_char, JobStatus.UNKNOWN)
    return result


# ---------------------------------------------------------------------------
# PBS scheduler
# ---------------------------------------------------------------------------

class PBSScheduler(Scheduler):
    """
    PBS/Torque scheduler backend.

    Submits via ``qsub``, polls via ``qstat``, cancels via ``qdel``.
    Supports both PBS Pro (JSON qstat) and Torque (tabular qstat).
    """

    renderer = PBSRenderer()

    def submit(self, script_path: Path) -> str:
        """
        Submit a job script via qsub.

        Returns
        -------
        str
            PBS job ID (e.g. "123456.server").  The ".server" suffix is
            stripped to return only the numeric part.

        Raises
        ------
        RuntimeError
            If qsub fails or returns unparseable output.
        """
        result = self._run(["qsub", str(script_path)])
        if result.returncode != 0:
            raise RuntimeError(
                f"qsub failed (exit {result.returncode}):\n{result.stderr.strip()}"
            )

        # qsub stdout: "123456.server" or just "123456"
        raw = result.stdout.strip()
        job_id = raw.split(".")[0]

        if not job_id.isdigit():
            raise RuntimeError(
                f"Could not parse job ID from qsub output: {raw!r}"
            )

        logger.info(f"  Submitted PBS job {job_id}: {script_path.name}")
        return job_id

    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """
        Query job statuses via qstat.

        Tries JSON output first (PBS Pro), falls back to tabular.
        Jobs not found are assumed done (aged out of scheduler memory).
        """
        if not job_ids:
            return {}

        # Try PBS Pro JSON format first
        result = self._run(["qstat", "-x", "-f", "-F", "json"] + job_ids)

        if result.returncode == 0 and result.stdout.strip().startswith("{"):
            statuses = _parse_qstat_json(result.stdout)
        else:
            # Fall back to plain qstat
            result = self._run(["qstat"] + job_ids)
            statuses = _parse_qstat_tabular(result.stdout) if result.returncode == 0 else {}

        for job_id in job_ids:
            if job_id not in statuses:
                statuses[job_id] = JobStatus.DONE

        return statuses

    def cancel(self, job_id: str) -> None:
        """Cancel a job via qdel."""
        result = self._run(["qdel", job_id])
        if result.returncode != 0:
            logger.warning(f"qdel {job_id} returned {result.returncode}: {result.stderr.strip()}")
        else:
            logger.info(f"  Cancelled PBS job {job_id}")
