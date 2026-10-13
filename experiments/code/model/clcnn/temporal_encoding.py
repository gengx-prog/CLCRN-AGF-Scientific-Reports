import math

import torch
import torch.nn as nn


class PeriodicTimeEncoder(nn.Module):
    """Encode calendar-style context into a compact periodic residual."""

    def __init__(
        self,
        output_dim,
        hidden_dim,
        use_year_trend=True,
        include_geo=False,
        year_base=2000.0,
        year_scale=20.0,
    ):
        super().__init__()
        self.use_year_trend = use_year_trend
        self.include_geo = include_geo
        self.year_base = year_base
        self.year_scale = year_scale

        input_dim = 6  # sin/cos of month, day, hour
        if self.use_year_trend:
            input_dim += 1
        if self.include_geo:
            input_dim += 4

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _sin_cos(value, period, offset=0.0):
        angle = 2.0 * math.pi * (value - offset) / period
        return torch.sin(angle), torch.cos(angle)

    def forward(self, context_inputs):
        if context_inputs.size(-1) < 4:
            raise ValueError("PeriodicTimeEncoder expects at least 4 context channels: year, month, day, hour.")

        context_inputs = context_inputs.float()
        year = context_inputs[..., 0:1]
        month = context_inputs[..., 1:2]
        day = context_inputs[..., 2:3]
        hour = context_inputs[..., 3:4]

        month_sin, month_cos = self._sin_cos(month - 1.0, period=12.0)
        day_sin, day_cos = self._sin_cos(day - 1.0, period=31.0)
        hour_sin, hour_cos = self._sin_cos(hour, period=24.0)

        pieces = [month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos]
        if self.use_year_trend:
            pieces.append((year - self.year_base) / self.year_scale)

        if self.include_geo:
            if context_inputs.size(-1) < 8:
                raise ValueError("PeriodicTimeEncoder with include_geo=True expects 8 context channels.")
            lsm = context_inputs[..., 4:5]
            height = torch.sign(context_inputs[..., 5:6]) * torch.log1p(context_inputs[..., 5:6].abs())
            lat = context_inputs[..., 6:7] / 90.0
            lon = context_inputs[..., 7:8] / 180.0
            pieces.extend([lsm, height, lat, lon])

        features = torch.cat(pieces, dim=-1)
        return self.proj(features)
