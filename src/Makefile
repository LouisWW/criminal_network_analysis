.EXPORT_ALL_VARIABLES:
notebooks_dir = $(shell find . -type f -name \*.ipynb ! -name '*checkpoint*')


.PHONY: compile_cython
compile_cython:
	@echo Compiling cython code
	python3 setup.py build_ext --inplace


.PHONY: test
test:
	@echo Running the tests with pytest
	python3 -m pytest

.PHONY: test_notebooks
test_notebooks:
	@echo Testing notebooks with pytest
	python3 -m pytest --nbmake $(notebooks)


.PHONY: docs_view
docs_view:
	@echo Creating API documentation
	pdoc3 --html config -o ./docs --force
	pdoc3 --html network_utils -o ./docs --force
	pdoc3 --html simulators -o ./docs -force
	pdoc3 --html test -o ./docs --force
	pdoc3 --html utils -o ./docs --force


.PHONY: profiling_sim_mart
profiling_sim_mart:
	python3 -m profile -o output.pstats main.py -read-data -sim-mart-vaq
	gprof2dot.py -f pstats output.pstats | dot -Tpng -o output.png



## Delete all compiled Python files
.PHONY: clean
clean:
	@echo Deleting all compiled files
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*.c" ! -name 'mt19937ar.c' -delete
	find . -type f -name "*.so" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .build
	rm -rf build
	rm -rf ../.myp_cache
	rm -rf rce
