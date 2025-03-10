# Installation Guide

## Basic Installation

```bash
pip install ocrtool
```

## Installation with Optional Components

```bash
# For tesseract support
pip install ocrtool[tesseract]

# For Google Document AI support
pip install ocrtool[gdai]

# For visualization tools
pip install ocrtool[visualization]

# For MIME type detection 
pip install ocrtool[magic]

# For all features
pip install ocrtool[all]
```

## System Dependencies

### libmagic Installation

The `magic` extra requires the libmagic system library. Installation varies by platform:

#### macOS:
```bash
brew install libmagic
```

#### Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y libmagic-dev
```

#### CentOS/RHEL/Fedora:
```bash
sudo yum install -y file-devel
```

#### Windows:

For Windows, follow instructions at:
https://github.com/ahupp/python-magic#dependencies

## Troubleshooting

If you encounter issues with python-magic after installation, ensure that:

1. The libmagic system library is properly installed
2. The `magic` module can find the libmagic library
3. You're using the correct version of python-magic for your OS