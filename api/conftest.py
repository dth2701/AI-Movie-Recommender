"""Put `api/` on sys.path so tests can `import train` from anywhere."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
