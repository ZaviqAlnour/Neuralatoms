"""
NeuralAtoms — Industrial Dependency Installer
Handles pip and system checks with 'skip if installed' logic.
"""

import subprocess
import sys
import os

def run_pip_install(package_list):
    print(f"Installing {len(package_list)} packages...")
    for pkg in package_list:
        try:
            # Check if installed
            clean_name = pkg.split('==')[0].split('>=')[0].split('<')[0].strip('"')
            subprocess.run([sys.executable, "-m", "pip", "show", clean_name], 
                           check=True, capture_output=True)
            print(f"  [SKIPPED] {pkg} already installed.")
        except subprocess.CalledProcessError:
            print(f"  [INSTALLING] {pkg}...")
            # Use --no-build-isolation for chumpy if needed, but let's try standard first
            cmd = [sys.executable, "-m", "pip", "install", pkg]
            if "chumpy" in pkg:
                # Chumpy is legacy, needs setuptools
                subprocess.run([sys.executable, "-m", "pip", "install", "setuptools"], check=True)
            
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  [FAILED] {pkg}: {res.stderr}")
            else:
                print(f"  [SUCCESS] {pkg}")

if __name__ == "__main__":
    packages = [
        "mujoco", "dm_control", "smplx", "trimesh", "pyyaml",
        "opencv-python-headless<4.9", "segment-anything-2", "depth-anything-v2"
    ]
    # Handle chumpy separately or with setuptools
    run_pip_install(packages)
    
    # Legacy Chumpy fix
    print("Applying Legacy Chumpy Fix...")
    subprocess.run([sys.executable, "-m", "pip", "install", "chumpy"], capture_output=True)
