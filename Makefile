
unit_tests_cpu:
	pytest -m "not requires_gpu" --durations 10 .

unit_tests:
	pytest --durations 10 .