"""
Integration test: run the doc-drift linter against the live repo.

This locks the whole codebase into the linter so that any PR which
edits settings.yaml without updating the tagged claims fails CI /
`pytest tests/ -v` immediately.
"""

from scripts.check_docs_consistency import run_check


def test_no_drift_in_tracked_claims():
    all_findings, failures = run_check()
    # If markers exist, make sure they all match. If nobody has tagged
    # anything yet, the test passes (nothing to compare) so we never
    # block on the one-time tagging chore.
    if not all_findings:
        return
    report = "\n".join(f.summary() for f in failures)
    assert not failures, (
        f"doc-drift detected: {len(failures)}/{len(all_findings)} tagged "
        f"claims disagree with config/settings.yaml:\n{report}"
    )
