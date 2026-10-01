"""Robust moving horizon estimation for agricultural vehicles with GNSS outliers.

Python + CasADi implementation of

    N. N. Deniz, G. M. Sanchez, F. A. Auat Cheein and L. L. Giovanini, "Robust Moving Horizon
    Estimation for Autonomous Agricultural Vehicles With GNSS Outliers Using a Robust Loss
    Function", IEEE Robotics and Automation Letters, 10(10):10815-10820, 2025.
    doi:10.1109/LRA.2025.3606377

The vehicle model, the path and the controller come from the nmppfc package
(github.com/nahueldeniz/nmppfc); this repository adds the loss, the redundant-coordinate
model with the chain equalities, and the estimator. See NOTES.md.
"""
from .config import BETA_MAX, Config, EstimatorConfig
from .estimator import EstimateInfo, RobustMHE, wrap
from .loss import garl, l2, psi, psi_symbolic, saturation, tail_exponent
from .model import RedundantNTrailer, VehicleGeometry

__all__ = ["BETA_MAX", "Config", "EstimatorConfig", "EstimateInfo", "RobustMHE", "wrap",
           "garl", "l2", "psi", "psi_symbolic", "saturation", "tail_exponent",
           "RedundantNTrailer", "VehicleGeometry"]
__version__ = "0.1.0"
