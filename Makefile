.PHONY: chapter01


repro:
	poetry run dvc repro

all: chapter01

dataset: data/raw/dataset.csv

data/raw/dataset.csv:
	mkdir -p data/raw/
	wget https://git.io/vXTVC -O data/raw/dataset.csv

chapter01: data/raw/dataset.csv
	poetry run python -m chapter01.messages_analysis data/raw/dataset.csv \
        --samples=40000 \
        --sample_tune=10000

lint:
	poetry run isort .
	poetry run black chapter* -l 79
	poetry run flake8 .
