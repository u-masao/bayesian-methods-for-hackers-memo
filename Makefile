.PHONY:

## dvc repro
repro: check_commit
	poetry run dvc repro || git commit dvc.lock -m '[update] dvc repro'
	poetry run dvc dag --md > PIPELINE.md
	git commit dvc.lock PIPELINE.md -m '[update] dvc repro' || true

## check commit
check_commit: lint
	git status
	git diff --exit-code --staged
	git diff --exit-code

## lint and format
lint:
	poetry run isort src
	poetry run black src -l 79
	poetry run flake8 src
