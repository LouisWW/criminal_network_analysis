
.PHONY: test
test:
	@echo Running the tests with pytest
	python3 -m pytest

.PHONY: docs_view
docs_view:
	@echo creating API documentation
	pdoc3 --html config -o ./docs --force
	pdoc3 --html network_utils -o ./docs --force
	pdoc3 --html simulators -o ./docs -force
	pdoc3 --html test -o ./docs --force
	pdoc3 --html utils -o ./docs --force

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*.c" -delete
	find . -type f -name "*.so" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .build
	rm -rf ../.myp_cache
