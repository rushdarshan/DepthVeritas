"""Pairwise ordinal depth supervision and metrics."""

from .ranking import ordinal_accuracy, ordinal_ranking_loss, sample_ordinal_pairs

__all__ = ["ordinal_accuracy", "ordinal_ranking_loss", "sample_ordinal_pairs"]
