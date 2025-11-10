# DataForSEO Project Makefile

.PHONY: help install test run clean lint format setup

help: ## Show this help message
	@echo "DataForSEO Project Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	pip install -r requirements.txt

setup: ## Setup project environment
	@echo "Setting up DataForSEO project..."
	@echo "1. Copy env.example to .env and fill in your credentials"
	@echo "2. Run 'make install' to install dependencies"
	@echo "3. Run 'make test' to verify setup"
	@echo "4. Run 'make run' to execute the pipeline"

test: ## Run tests
	pytest src/tests/ -v

run: ## Run the Kedro pipeline
	kedro run

run-dry: ## Run pipeline in dry-run mode
	kedro run --dry-run

viz: ## Launch Kedro Viz
	kedro viz

lint: ## Run linting
	flake8 src/
	black --check src/

format: ## Format code
	black src/
	isort src/

clean: ## Clean temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

clean-data: ## Clean data directories (WARNING: removes all data)
	rm -rf data/02_intermediate/
	rm -rf data/03_model_input/
	rm -rf data/04_model/
	rm -rf data/05_reporting/

clean-cache: ## Clean cache files
	rm -rf data/02_intermediate/cache/
	find . -name "*.cache" -delete

check-env: ## Check environment variables
	@echo "Checking environment variables..."
	@python -c "import os; print('DATAFORSEO_USERNAME:', 'SET' if os.getenv('DATAFORSEO_USERNAME') else 'NOT SET'); print('DATAFORSEO_PASSWORD:', 'SET' if os.getenv('DATAFORSEO_PASSWORD') else 'NOT SET'); print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"

validate: ## Validate data and configuration
	@echo "Validating project configuration..."
	@python -c "from src.dataforseo_project.utils.data_quality import generate_data_quality_report; print('Data quality utilities loaded successfully')"

docs: ## Generate documentation
	@echo "Generating documentation..."
	@echo "Documentation would be generated here"

all: install test lint ## Install, test, and lint
