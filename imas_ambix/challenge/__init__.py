"""Sophelio challenge corpus acquisition and canonical typed loading."""

from .convention import DIIID_CONVENTION, DIIID_SOURCE_COCOS
from .loader import ChallengeShot, EfitLabels, SignalSeries, ThomsonProfile, load_shot

__all__ = [
    "DIIID_CONVENTION",
    "DIIID_SOURCE_COCOS",
    "ChallengeShot",
    "EfitLabels",
    "SignalSeries",
    "ThomsonProfile",
    "load_shot",
]
