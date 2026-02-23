"""
gocia.calculator

Staged optimisation pipeline and calculator backends.

Submodules
----------
stage       CalculatorStage dataclass and pipeline builder
pipeline    Run stages in order, manage sentinels, write trajectory.h5
mace_calc   MACE-MP-0 universal potential backend
vasp_calc   VASP backend via ASE, with default INCAR + per-stage overrides

Typical usage
-------------
    from gocia.calculator.stage import build_pipeline
    from gocia.calculator.pipeline import run_pipeline
    from gocia.config import load_config

    cfg = load_config("gocia.yaml")
    stages = build_pipeline(cfg.calculator_stages)
    relaxed, energy = run_pipeline(atoms, stages, struct_dir)
"""
