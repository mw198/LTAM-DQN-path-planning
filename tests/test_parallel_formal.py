from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiments import resolve_methods
from src.merge_results import _copy_tree_contents, list_eval_metric_files


def test_resolve_methods_filters_subset():
    selected = resolve_methods("DQN,LTAM-DQN")
    assert selected == ["DQN", "LTAM-DQN"]


def test_list_eval_metric_files_discovers_all_files(tmp_path: Path):
    file_a = tmp_path / "a" / "metrics_csv" / "E0-main" / "DQN" / "seed_0" / "eval_metrics.csv"
    file_b = tmp_path / "b" / "metrics_csv" / "E1-large" / "LTAM-DQN" / "seed_1" / "eval_metrics.csv"
    file_a.parent.mkdir(parents=True)
    file_b.parent.mkdir(parents=True)
    file_a.write_text("method,success\nDQN,1\n", encoding="utf-8")
    file_b.write_text("method,success\nLTAM-DQN,1\n", encoding="utf-8")

    files = list_eval_metric_files([tmp_path / "a", tmp_path / "b"])
    assert len(files) == 2


def test_copy_tree_contents_tolerates_same_source_and_target(tmp_path: Path):
    root = tmp_path / "root"
    file_path = root / "configs" / "a.json"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("{}", encoding="utf-8")

    _copy_tree_contents(root, root)
    assert file_path.exists()
