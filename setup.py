from typing import Mapping, Type

from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
import platform
import subprocess
import sys


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

# Core dependencies that are always needed
CORE_DEPS = [
    "pillow>=11.1.0",  # Required for image processing
    "python-dstools>=0.1.4",
    "PyPDF2>=3.0.1",  # Required for PDF splitting and page counting
    "numpy>=1.26.4",  # Required for basic array operations
]

# Optional dependencies for specific features
EXTRAS = {
    "tesseract": [
        "pytesseract>=0.3.13",
    ],
    "gdai": [
        "google-cloud-documentai>=3.4.0",
        "google-auth>=2.39.0",
        "numpy~=1.26.4"
    ],
    "dev": [
        "pytest>=8.3.5",
        "flake8>=7.2.0",
        "mypy>=25.1.0",
        "black>=25.1.0",
    ],
    "visualization": [
        "matplotlib>=3.10.1",
    ],
    "magic": [
        "python-magic>=0.4.27",  # For MIME type detection
    ],
    "scan": [
        "opencv-python>=4.11.0.86",
    ],
}

# Add the 'all' extra that includes everything except 'dev'
ALL_EXTRAS = []
for extra in EXTRAS.keys():
    if extra != "dev":
        ALL_EXTRAS.extend(EXTRAS[extra])

EXTRAS["all"] = list(set(ALL_EXTRAS))  # Remove any duplicates


class CustomEggInfo(egg_info):
    """Custom egg_info command that adds 'all' extras to install_requires."""
    
    def run(self):
        """Run the command, adding 'all' extras to install_requires."""
        if self.distribution.install_requires:
            self.distribution.install_requires.extend(EXTRAS['all'])
        else:
            self.distribution.install_requires = EXTRAS['all']
        super().run()


setup(
    name="ocr-compare",
    version="0.2.2",
    description="A tool for comparing OCR results from different OCR engines",
    author="OCR Compare Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=CORE_DEPS,
    extras_require=EXTRAS,
    scripts=[
        "examples/gdai_example.py",
        "examples/ocr_comparison.py",
        "examples/basic_usage.py",
    ],
    # Make 'all' the default installation
    cmdclass={
        'egg_info': CustomEggInfo,
    }
)


