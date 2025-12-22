#* Variables
SHELL := /usr/bin/env bash
PYTHON := python
OS := $(shell python -c "import sys; print(sys.platform)")

ifeq ($(OS),win32)
	PYTHONPATH := $(shell python -c "import os; print(os.getcwd())")
    TEST_COMMAND := set PYTHONPATH=$(PYTHONPATH) && uv run pytest -c pyproject.toml --cov-report=html --cov=molx_agent tests/
else
	PYTHONPATH := `pwd`
    TEST_COMMAND := PYTHONPATH=$(PYTHONPATH) uv run pytest -c pyproject.toml --cov-report=html --cov=molx_agent tests/
endif

#* Docker variables
IMAGE := molx_agent
VERSION := latest

.PHONY: lock install pre-commit-install polish-codestyle formatting test check-codestyle lint docker-build docker-remove cleanup help

lock:
	uv lock

install:
	uv sync --extra dev --extra server

pre-commit-install:
	uv run pre-commit install

polish-codestyle:
	uv run ruff format --config pyproject.toml .
	uv run ruff check --fix --config pyproject.toml .

formatting: polish-codestyle
format: polish-codestyle

test:
	$(TEST_COMMAND)
	uv run coverage-badge -o assets/images/coverage.svg -f

check-codestyle:
	uv run ruff format --check --config pyproject.toml .
	uv run ruff check --config pyproject.toml .

check-safety:
	uv run safety check --full-report
	uv run bandit -ll --recursive molx_agent tests

lint: test check-codestyle check-safety

# Example: make docker-build VERSION=latest
# Example: make docker-build IMAGE=some_name VERSION=0.1.0
docker-build:
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build \
		-t $(IMAGE):$(VERSION) . \
		-f ./docker/Dockerfile --no-cache

# Example: make docker-remove VERSION=latest
# Example: make docker-remove IMAGE=some_name VERSION=0.1.0
docker-remove:
	@echo Removing docker $(IMAGE):$(VERSION) ...
	docker rmi -f $(IMAGE):$(VERSION)

cleanup:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf
	find . | grep -E ".DS_Store" | xargs rm -rf
	find . | grep -E ".mypy_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	rm -rf build/

serve:
	uv run molx-server run

run-client:
	cd molx_client && npm run dev

help:
	@echo "lock                                      Lock the dependencies."
	@echo "install                                   Install the project dependencies."
	@echo "pre-commit-install                        Install the pre-commit hooks."
	@echo "polish-codestyle                          Format the codebase."
	@echo "formatting                                Format the codebase."
	@echo "test                                      Run the tests."
	@echo "check-codestyle                           Check the codebase for style issues."
	@echo "lint                                      Run the tests and check the codebase for style issues."
	@echo "docker-build                              Build the docker image."
	@echo "docker-remove                             Remove the docker image."
	@echo "cleanup                                   Clean the project directory."
	@echo "help                                      Display this help message."
