"""
The `pystim.datasets` module includes utilities to load datasets,
including methods to load and fetch datasets.
"""

from .base import get_path
from .van_hateren import fetch as fetch_van_hateren


__all__ = [
    'fetch_van_hateren',
    'get_path',
]
