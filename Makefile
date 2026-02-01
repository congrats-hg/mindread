.PHONY: install install-dev test lint format clean train eval demo help

help:
	@echo "MindRead - Dialogue State Tracking"
	@echo ""
	@echo "Usage:"
	@echo "  make install      Install package dependencies"
	@echo "  make install-dev  Install with development dependencies"
	@echo "  make test         Run tests with coverage"
	@echo "  make lint         Run linting checks"
	@echo "  make format       Format code with ruff"
	@echo "  make clean        Remove build artifacts"
	@echo "  make train        Train the model"
	@echo "  make eval         Evaluate the model"
	@echo "  make demo         Launch Gradio demo"
	@echo "  make data         Download and prepare DSTC2 data"

install:
	pip install -e .

install-dev:
	pip install -e ".[all]"

test:
	pytest tests/ -v --cov=mindread --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

train:
	python -m mindread.training.train

eval:
	python -m mindread.evaluation.evaluate

demo:
	python -m mindread.demo

data:
	python -m mindread.data.download
