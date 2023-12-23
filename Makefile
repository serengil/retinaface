test:
	python -m pytest tests/ -s --disable-warnings

lint:
	python -m pylint retinaface/ --fail-under=10