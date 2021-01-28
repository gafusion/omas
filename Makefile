VERSION := $(shell cat omas/version)

all:
	@echo 'OMAS $(VERSION) makefile help'
	@echo ''
	@echo ' - make tests         : run all regression tests'
	@echo ' - make omfit_tests   : run test_omas in OMFIT'
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
	@echo ' - make format        : format source using black'
	@echo ' - make site-packages : pip install requirements in site-packages folder'
	@echo ''

TEST_FLAGS=-s omas/tests -v -f

test:
	python3 -m unittest discover --pattern="*.py" ${TEST_FLAGS}

test_core:
	python3 -m unittest discover --pattern="*_core.py" ${TEST_FLAGS}

test_plot:
	python3 -m unittest discover --pattern="*_plot.py" ${TEST_FLAGS}

test_physics:
	python3 -m unittest discover --pattern="*_physics.py" ${TEST_FLAGS}

test_machine:
	python3 -m unittest discover --pattern="*_machine.py" ${TEST_FLAGS}

test_utils:
	python3 -m unittest discover --pattern="*_utils.py" ${TEST_FLAGS}

test_examples:
	python3 -m unittest discover --pattern="*_examples.py" ${TEST_FLAGS}

test_suite:
	python3 -m unittest discover --pattern="*_suite.py" ${TEST_FLAGS}

requirements:
	rm -f requirements.txt
	python3 setup.py --name

html:
	cd sphinx && make html

examples:
	cd sphinx && make examples

samples:
	cd omas/utilities && python3 generate_ods_samples.py

docs: html
	cd sphinx && make commit && make push

json:
	cd omas/utilities && python3 build_json_structures.py
	make cocos

cocos: symbols
	cd omas/utilities && python3 generate_cocos_signals.py

machines:
	cd omas/utilities && python3 format_machine_mappings.py

symbols:
	cd omas/utilities && python3 sort_symbols.py

tag:
	git tag -a v$(VERSION) $$(git log --pretty=format:"%h" --grep="^version $(VERSION)") -m "version $(VERSION)"
	git push --tags

sdist:
	rm -rf dist
	python3 setup.py sdist

pypi: sdist
	python3 -m twine upload --repository pypi dist/omas-$(VERSION).tar.gz

testpypi:
	python3 -m twine upload --repository testpypi dist/omas-$(VERSION).tar.gz
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
