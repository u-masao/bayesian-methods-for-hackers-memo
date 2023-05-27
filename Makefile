.PHONY: chapter01

all: chapter01

chapter01:
	poetry run python -m chapter01.messages_analysis

lint:
	poetry run isort .
	poetry run black chapter* -l 79
	poetry run flake8 .
