from __future__ import annotations

import math

import torch

from .visual_multiplace_aba import VisualMultiPlaceConfig, VisualMultiPlaceReMAPFormer
from .visual_simulator_aba import SimulatorABADataset


class SimulatorCoupledWaypointReMAPFormer(VisualMultiPlaceReMAPFormer):
    """Translated waypoint model with fixed simulator-calibrated action scales."""

    def __init__(
        self,
        config: VisualMultiPlaceConfig = VisualMultiPlaceConfig(
            max_sequence_length=384,
            state_window_size=128,
        ),
    ) -> None:
        super().__init__(config)
        with torch.no_grad():
            self.ec.log_forward_scale.fill_(math.log(0.25))
            self.ec.log_turn_scale.fill_(math.log(math.pi / 10.0))
        self.ec.log_forward_scale.requires_grad_(False)
        self.ec.log_turn_scale.requires_grad_(False)


SimulatorWaypointABADataset = SimulatorABADataset
