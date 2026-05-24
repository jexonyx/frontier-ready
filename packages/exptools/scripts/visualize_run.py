#!/usr/bin/env python3
"""CLI wrapper for visualization module."""

import sys
from pathlib import Path

# Allow running from any location
sys.path.insert(0, str(Path(__file__).parent.parent))

from exptools.visualization import main

if __name__ == "__main__":
    main()
