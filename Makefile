
unit_tests_cpu:
	pytest -m "not requires_gpu" --durations 10 .

unit_tests:
	pytest --durations 10 .

inference:
	python -m "fme.fcn_training.inference.inference" "./config/explore-inference.yaml"

fmace-inference:
	python -m "fmace.pipeline.inference" "./config/explore-inference.yaml"
