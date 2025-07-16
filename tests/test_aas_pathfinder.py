import json
from pathlib import Path

import aas_pathfinder as ap


def test_candidate_selection(tmp_path: Path) -> None:
    data = [
        {"name": "AAS_1000ton", "status": "active", "score": 1},
        {"name": "AAS_2000ton", "status": "active", "score": 3},
        {"name": "AAS_3000ton", "status": "inactive", "score": 5},
    ]
    for i, d in enumerate(data):
        (tmp_path / f"f{i}.json").write_text(json.dumps(d))

    aas_list = ap.load_aas_files(str(tmp_path))
    candidates = ap.filter_candidates(aas_list, status="active")
    best = ap.choose_best_candidate(candidates)
    assert best["name"] == "AAS_2000ton"


def test_visualize_flow(tmp_path: Path) -> None:
    aas_list = [{"name": "AAS_1"}]
    out_file = tmp_path / "out.html"
    ap.visualize_flow(aas_list, str(out_file))
    assert out_file.exists() and out_file.stat().st_size > 0
