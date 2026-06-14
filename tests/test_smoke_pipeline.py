from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import PROJECT_ROOT as APP_ROOT, load_yaml
from src.run_experiments import run_sanity_experiment


def test_default_config_loads():
    config_path = APP_ROOT / "configs" / "default.yaml"
    data = load_yaml(config_path)
    assert data["env"]["map_size"] == 15
    assert data["train"]["batch_size"] > 0
    assert "LTAM-DQN" in data["methods"]["formal"]


def test_run_sanity_experiment_writes_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir) / "results"
        run_sanity_experiment(project_root=APP_ROOT, output_root=output_root, episodes=5, seed=0)
        assert (output_root / "logs").exists()
        assert any(output_root.rglob("*.csv"))
