"""Confidence-gated physical depth priors."""

from .plane import fit_plane_from_depth, planar_huber_loss

__all__ = ["fit_plane_from_depth", "planar_huber_loss"]
