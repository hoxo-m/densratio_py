init:
	pip install -r requirements.txt

test:
	pytest

docs:
	cd docs && make html
	@echo "\033[95m\n\nBuild successful! View the docs homepage at docs/_build/html/index.html.\n\033[0m"

install:
	python setup.py install
