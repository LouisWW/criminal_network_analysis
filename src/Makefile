
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
