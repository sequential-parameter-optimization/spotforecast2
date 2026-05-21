# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Training driver for spotforecast2.

The full-training entry point lives in `spotforecast2.trainer.trainer_full`.
Multi-target task classes have moved to `spotforecast2.multitask`, and the
full-featured forecaster classes have moved to `spotforecast2.models`;
import them from there directly — they are not re-exported here.
"""
