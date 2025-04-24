from setuptools import setup, find_packages
import platform
import subprocess
import sys
import os


def install_libmagic():
    """Install libmagic based on the operating system."""
    system = platform.system().lower()

    if system == "darwin":  # macOS
        try:
            # Check if brew is available
            subprocess.run(["brew", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Installing libmagic via Homebrew...")
            subprocess.run(["brew", "install", "libmagic"], check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Homebrew not found. Please install libmagic manually:")
            print("    brew install libmagic")

    elif system == "linux":
        # Check for different Linux distributions
        try:
            with open("/etc/os-release") as f:
                os_info = f.read()

            if "debian" in os_info.lower() or "ubuntu" in os_info.lower():
                print("Installing libmagic via apt...")
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "libmagic-dev"], check=True)

            elif "fedora" in os_info.lower() or "centos" in os_info.lower() or "rhel" in os_info.lower():
                print("Installing libmagic via yum...")
                subprocess.run(["yum", "install", "-y", "file-devel"], check=True)

            else:
                print("Unsupported Linux distribution. Please install libmagic manually.")
        except (subprocess.SubprocessError, FileNotFoundError, IOError):
            print("Unable to determine Linux distribution. Please install libmagic manually.")

    elif system == "windows":
        print("For Windows, please follow the instructions at:")
        print("https://github.com/ahupp/python-magic#dependencies")

    else:
        print(f"Unsupported operating system: {system}. Please install libmagic manually.")


# Install libmagic if the magic extra is requested
if any('magic' in arg or 'all' in arg for arg in sys.argv):
    install_libmagic()

setup(
    name="ocr-compare",
    version="0.2.2",
    description="A tool for comparing OCR results from different OCR engines",
    author="OCR Compare Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "pillow",  # Required for image processing,
        "python-dstools",
        "PyPDF2",  # Required for PDF splitting and page counting
    ],
    extras_require={
        "tesseract": ["pytesseract"],
        "gdai": [
            "google-cloud-documentai",
            "google-auth",
        ],
        "dev": [
            "pytest",
            "flake8",
            "mypy",
            "black",
        ],
        "visualization": [
            "matplotlib",
            "numpy",
        ],
        "magic": [
            "python-magic",  # For MIME type detection
        ],
        "scan": [
            "opencv-python",
            "numpy",
        ],
        "all": [
            "pytesseract",
            "google-cloud-documentai",
            "google-auth",
            "matplotlib",
            "numpy",
            "python-magic",
            "pandas",
            "opencv-python",
            "PyPDF2",
        ],
    },
    scripts=[
        "examples/gdai_example.py",
        "examples/ocr_comparison.py",
        "examples/basic_usage.py",
    ],
)
