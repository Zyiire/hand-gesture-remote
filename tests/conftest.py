import sys
from pathlib import Path

# Make the project root (gesture_remote) and this folder (synthetic) importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
