VERSION := $(shell cat omas/version)

all:
	@echo 'OMAS $(VERSION) makefile help'
	@echo ''
	@echo ' - make test          : run all regression tests'
	@echo ' - make requirements  : build requirements.txt'
	@echo ' - make json          : generate IMAS json structure files'
	@echo ' - make docs          : generate sphinx documentation and pushes it online'
	@echo ' - make tag           : tag git repository with omas/version and push'
	@echo ' - make cocos         : generate list of COCOS transformations'
	@echo ' - make machines      : format machine mappings files'
	@echo ' - make release       : all of the above, in order'
	@echo ' - make pypi          : upload to pypi'
	@echo ' - make html          : generate sphinx documentation'
	@echo ' - make examples      : generate sphinx documentation with examples'
	@echo ' - make samples       : generate sample files'
	@echo ' - make omfit         : scan OMFIT to_omas and from_omas for documentation'
	@echo ' - make format        : format source using black'
	@echo ' - make site-packages : pip install requirements in site-packages folder'
	@echo ''

TEST_FLAGS=-f -b -v -c -s omas/tests

OMAS_PYTHON ?= python3

test:
	${OMAS_PYTHON} -m unittest discover --pattern="*.py" ${TEST_FLAGS}

test_core:
	${OMAS_PYTHON} -m unittest discover --pattern="*_core.py" ${TEST_FLAGS}

test_plot:
	${OMAS_PYTHON} -m unittest discover --pattern="*_plot.py" ${TEST_FLAGS}

test_physics:
	${OMAS_PYTHON} -m unittest discover --pattern="*_physics.py" ${TEST_FLAGS}

test_fakeimas:
	${OMAS_PYTHON} -m unittest discover --pattern="*_fakeimas.py" ${TEST_FLAGS}

test_machine:
	${OMAS_PYTHON} -m unittest discover --pattern="*_machine.py" ${TEST_FLAGS}

test_utils:
	${OMAS_PYTHON} -m unittest discover --pattern="*_utils.py" ${TEST_FLAGS}

test_examples:
	${OMAS_PYTHON} -m unittest discover --pattern="*_examples.py" ${TEST_FLAGS}

test_suite:
	${OMAS_PYTHON} -m unittest discover --pattern="*_suite.py" ${TEST_FLAGS}

test_no_munittest:
	omas/tests/run_tests.sh

requirements:
	rm -f requirements.txt
	${OMAS_PYTHON} setup.py --name

omfit:
	cd omas/utilities && ${OMAS_PYTHON} generate_to_from_omas.py

html:
	cd sphinx && make html

examples:
	cd sphinx && make examples

samples:
	cd omas/utilities && ${OMAS_PYTHON} generate_ods_samples.py

docs: html
	cd sphinx && make commit && make push && rm -rf build

json:
	cd omas/utilities && ${OMAS_PYTHON} build_json_structures.py
	make cocos

cocos: symbols
	cd omas/utilities && ${OMAS_PYTHON} generate_cocos_signals.py

machines:
	cd omas/utilities && ${OMAS_PYTHON} format_machine_mappings.py

symbols:
	cd omas/utilities && ${OMAS_PYTHON} sort_symbols.py

tag:
	git tag -a v$(VERSION) $$(git log --pretty=format:"%h" --grep="^version $(VERSION)") -m "version $(VERSION)"
	git push --tags

sdist:
	rm -rf dist
	${OMAS_PYTHON} setup.py sdist

pypi: sdist
	${OMAS_PYTHON} -m twine upload --repository pypi dist/omas-$(VERSION).tar.gz

testpypi:
	${OMAS_PYTHON} -m twine upload --repository testpypi dist/omas-$(VERSION).tar.gz
	@echo install with:
	@echo pip install --index-url https://test.pypi.org/simple/ omas

hash:
	pip hash dist/omas-$(VERSION).tar.gz

release: tests requirements json cocos docs tag
	@echo 'Make release done'

fmt:
	black -S -l 140 .

format:fmt cocos machines

.PHONY: site-packages

site-packages:
	pip install --upgrade --target ./site-packages -r requirements.txt
	@echo "for TCSH: setenv PYTHONPATH $$PWD/site-packages:\$$PYTHONPATH"
	@echo "for BASH: export PYTHONPATH=$$PWD/site-packages:\$$PYTHONPATH"
