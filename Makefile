all:
	@echo 'OMAS makefile help'
	@echo ''
	@echo ' - make tests2       : run all regression tests with Python2'
	@echo ' - make tests3       : run all regression tests with Python3'
	@echo ' - make requirements : build requirements.txt'
	@echo ' - make json         : generate IMAS json structure files'
	@echo ' - make docs         : generate sphinx documentation and pushes it online'
	@echo ' - make tag          : tag git repository with omas/version and push'
	@echo ' - make cocos        : generate list of COCOS transformations'
	@echo ' - make release      : all of the above, in order'
	@echo ' - make pypi         : upload to pypi'
	@echo ' - make html         : generate sphinx documentation'
	@echo ' - make examples     : generate sphinx documentation with examples'
	@echo ''

tests: tests2 tests3

TEST_FLAGS=-s omas/tests -v -f

tests2:
	python2 -m unittest discover --pattern="*.py" ${TEST_FLAGS}

tests3:
	python3 -m unittest discover --pattern="*.py" ${TEST_FLAGS}

tests_plot: tests2_plot tests3_plot

tests2_plot:
	python2 -m unittest discover --pattern="*_plot.py" ${TEST_FLAGS}

tests3_plot:
	python3 -m unittest discover --pattern="*_plot.py" ${TEST_FLAGS}

tests_physics: tests2_physics tests3_physics

tests2_physics:
	python2 -m unittest discover --pattern="*_physics.py" ${TEST_FLAGS}

tests3_physics:
	python3 -m unittest discover --pattern="*_physics.py" ${TEST_FLAGS}

tests_utils: tests2_utils tests3_utils

tests2_utils:
	python2 -m unittest discover --pattern="*_utils.py" ${TEST_FLAGS}

tests3_utils:
	python3 -m unittest discover --pattern="*_utils.py" ${TEST_FLAGS}

requirements:
	rm -f requirements_python2.txt
	rm -f requirements_python3.txt
	python setup.py --name

html:
	cd sphinx && make html

examples:
	cd sphinx && make examples

docs: html
	cd sphinx && make commit && make push

json:
	cd omas/utilities && python build_json_structures.py
	make cocos

cocos:
	cd omas/utilities && python generate_cocos_signals.py

tag:
	git tag -a v$$(cat omas/version) $$(git log --pretty=format:"%h" --grep="^version $$(cat omas/version)") -m "version $$(cat omas/version)"
	git push --tags

pypi:
	python setup.py sdist upload

release: tests2 tests3 requirements json cocos docs tag
	@echo 'Make release done'
