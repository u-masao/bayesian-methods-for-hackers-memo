.PHONY: chapter01

all: chapter01

dataset: data/raw/dataset.csv

data/raw/dataset.csv:
	mkdir -p data/raw/
	wget https://git.io/vXTVC -O data/raw/dataset.csv

chapter01: data/raw/dataset.csv
	poetry run python -m chapter01.messages_analysis data/raw/dataset.csv

lint:
	poetry run isort .
	poetry run black chapter* -l 79
	poetry run flake8 .
