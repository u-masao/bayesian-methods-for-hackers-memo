.PHONY:

## dvc repro
repro: check_commit PIPELINE.md
	poetry run dvc repro || git commit dvc.lock -m '[update] dvc repro'
	git commit dvc.lock PIPELINE.md -m '[update] dvc repro' || true

## make pipeline markdown
PIPELINE.md: dvc.yaml params.yaml
	echo "## summary" > PIPELINE.md
	poetry run dvc dag --md >> PIPELINE.md
	echo "## detail" >> PIPELINE.md
	poetry run dvc dag --md -o >> PIPELINE.md

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
