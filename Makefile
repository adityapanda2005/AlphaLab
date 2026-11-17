.PHONY: install test lint pipeline clean docker

install:
	pip install -e ".[all]"

test:
	pytest tests/ --tb=short -q

lint:
	ruff check src/ tests/

pipeline:
	python -m alphalab.pipeline

pipeline-realdata:
	python -m alphalab.pipeline --source yfinance

clean:
	rm -rf data/cache/ artifacts/ dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

docker:
	docker build -t alphalab .
	docker run --rm alphalab
