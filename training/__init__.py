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

__all__ = ["Trainer"]


def __getattr__(name):
    """Keep lightweight training utilities importable without trainer extras.

    Importing ``training.score_se3_metrics`` must not eagerly import the full
    data/trainer stack (WebDataset, Numba, W&B, CUDA extensions).  Existing
    ``from training import Trainer`` callers retain the same public API.
    """

    if name == "Trainer":
        from training.trainer import Trainer

        return Trainer
    raise AttributeError(name)
