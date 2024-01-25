
unit_tests:
	pytest -m "not requires_gpu" --durations 10 .