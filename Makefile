# Name of the Conda environment
ENV_NAME = symbac_development

# Python version to use in the Conda environment
PYTHON_VERSION = 3.12

# Name of the package
PACKAGE_NAME = symbac

.PHONY: all create-env install uninstall clean reinstall activate

all: reinstall

# Create a new Conda environment
create-env:
	conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y

# Install the package in editable mode along with development and GPU dependencies
install:
	conda activate $(ENV_NAME) && \
	pip install -e .[dev,gpu]

# Uninstall the package
uninstall:
	conda activate $(ENV_NAME) && \
	pip uninstall -y $(PACKAGE_NAME)

# Clean up build files and caches
clean:
	rm -rf build/ dist/ $(PACKAGE_NAME).egg-info && \
	pip cache purge

# Reinstall the package after cleaning
reinstall: uninstall clean install

# Activate the Conda environment
activate:
	conda activate $(ENV_NAME)
