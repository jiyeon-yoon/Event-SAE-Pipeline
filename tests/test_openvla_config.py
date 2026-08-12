import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "event_sae_openvla_eval_config",
    ROOT / "event_sae/openvla/eval/config.py",
)
assert SPEC is not None and SPEC.loader is not None
CONFIG_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONFIG_MODULE
SPEC.loader.exec_module(CONFIG_MODULE)
load_config = CONFIG_MODULE.load_config


def test_reproduction_config_pins_weights_and_remote_code():
    cfg = load_config(
        ROOT / "configs/reproduction/openvla/libero_spatial_hooked_sr_layer31.yaml"
    )
    assert cfg.model.revision == "962318cec55ac10993ff0f5f43eda9a270b4c873"
    assert cfg.model.code_revision == "47a0ec7fc4ec123775a391911046cf33cf9ed83f"


def test_reproduction_rollout_budgets_match_paper_protocol():
    hooked = load_config(
        ROOT / "configs/reproduction/openvla/libero_spatial_hooked_sr_layer31.yaml"
    )
    intervention = load_config(
        ROOT / "configs/reproduction/openvla/libero_spatial_intervention_layer31.yaml"
    )
    assert hooked.env.num_trials_per_task == 10
    assert intervention.env.num_trials_per_task == 50
