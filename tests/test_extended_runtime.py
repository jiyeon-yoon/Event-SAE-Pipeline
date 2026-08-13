from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.runtime import to_executed_libero_action


def test_raw_action_is_preserved_while_gripper_convention_is_converted():
    raw = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0])
    executed = to_executed_libero_action(raw)
    np.testing.assert_array_equal(raw, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0])
    np.testing.assert_array_equal(executed, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0])
