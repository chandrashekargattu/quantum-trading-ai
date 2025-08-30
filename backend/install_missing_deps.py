#!/usr/bin/env python3
"""Install missing dependencies for the backend"""

import subprocess
import sys

MISSING_DEPS = [
    "openai",
    "transformers",
    "huggingface-hub",
    "tokenizers==0.21.0",
    "regex",
    "bsedata",
    "kiteconnect",
    "nsetools",
    "torch",
    "torchvision",
    "opencv-python",
    "rasterio",
    "geopandas",
    "shapely",
    "vaderSentiment",
    "arch",
    "copulas",
    "sortedcontainers",
    "numba",
    "pymongo",
]

def install_deps():
    """Install missing dependencies"""
    print("📦 Installing missing dependencies...")
    
    for dep in MISSING_DEPS:
        print(f"\n🔧 Installing {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {dep}, continuing...")
    
    print("\n✅ Dependency installation complete!")

if __name__ == "__main__":
    install_deps()

