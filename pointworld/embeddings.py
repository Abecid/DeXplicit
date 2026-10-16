# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn


def posemb_sincos_torch(pos, dim, min_period=4e-3, max_period=4.0):
    if dim % 2:
        raise ValueError("embed dim must be even")
    device = pos.device
    half = dim // 2
    frac = torch.linspace(0, 1, half, device=device)
    period = min_period * (max_period / min_period) ** frac
    sinusoid_input = pos.unsqueeze(-1) / period * 2 * math.pi
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)


class TemporalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t_scalar):
        emb = posemb_sincos_torch(t_scalar, self.embed_dim)
        return self.mlp(emb)


class SpatialEmbedding(nn.Module):
    def __init__(self, embed_dim, num_frequencies=None, min_period=4e-3, max_period=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies or max(1, embed_dim // 6)
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        raw_dim = 3 * 2 * self.num_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, coord):
        if coord.shape[-1] != 3:
            raise ValueError(f"SpatialEmbedding expects xyz coordinates, got shape {tuple(coord.shape)}")
        device = coord.device
        frac = torch.linspace(0, 1, self.num_frequencies, device=device, dtype=coord.dtype)
        period = self.min_period * (self.max_period / self.min_period) ** frac
        sinusoid_input = coord.unsqueeze(-1) / period * 2 * math.pi
        emb = torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)
        emb = emb.flatten(start_dim=-2)
        return self.mlp(emb)
