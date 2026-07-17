import os
from pathlib import Path

from examples.stochastic_sppt.figures import run_convergence


def test_convergence_smoke_and_resume(tmp_path, monkeypatch):
    # Point artifacts at a temp dir so the test is hermetic.
    monkeypatch.setattr(run_convergence, "ART", tmp_path)
    monkeypatch.setattr(run_convergence, "DATASET", tmp_path / "ds.pkl")
    monkeypatch.setattr(run_convergence, "CKPT", tmp_path / "ck.pkl")

    argv = ["--cases", "2", "--draws", "2", "--members", "2", "--batch", "2",
            "--target-steps", "1", "--lead-days", "1", "--spinup-days", "1",
            "--budget-seconds", "600"]
    run_convergence.main(argv)
    assert (tmp_path / "ck.pkl").exists()
    assert (tmp_path / "ds.pkl").exists()

    # Re-invoking with a higher target resumes from the checkpoint (step advances).
    import pickle
    with open(tmp_path / "ck.pkl", "rb") as f:
        step_after_first = pickle.load(f)["step"]
    assert step_after_first == 1
    run_convergence.main(argv[:-2] + ["--target-steps", "2", "--budget-seconds", "600"])
    with open(tmp_path / "ck.pkl", "rb") as f:
        assert pickle.load(f)["step"] == 2
