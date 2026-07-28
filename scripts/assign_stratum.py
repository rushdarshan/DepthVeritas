#!/usr/bin/env python3
"""Deterministically assign benchmark strata from curated metadata tags."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from depthlab.benchmark import PRIMARY_STRATA


def assign_stratum(tags: Iterable[str]) -> Dict[str, Any]:
    normalized = {tag.lower().replace(" ", "_") for tag in tags}
    for stratum, sub_strata in PRIMARY_STRATA.items():
        for sub_stratum in sub_strata:
            if sub_stratum in normalized:
                return {"stratum": stratum, "sub_stratum": sub_stratum, "needs_human_review": False}
    return {"stratum": "unknown", "sub_stratum": "unknown", "needs_human_review": True}
