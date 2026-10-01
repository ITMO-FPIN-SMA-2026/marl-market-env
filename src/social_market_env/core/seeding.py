"""Independent integer seeds derived from one seed with numpy's SeedSequence."""

import numpy as np


def spawn_seeds(*, seed: int, count: int) -> list[int]:
    """Return count statistically independent integer seeds derived from seed."""
    return [int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(count)]
