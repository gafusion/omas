all:
	@echo 'OMAS makefile help'
	@echo ''
	@echo ' - make tests2       : run all regression tests with Python2'
	@echo ' - make tests3       : run all regression tests with Python3'
	@echo ' - make requirements : build requirements.txt'
	@echo ' - make json         : generate IMAS json structure files'
	@echo ' - make itm          : generate omas_itm.py from omas_imas.py'
	@echo ' - make docs         : generate sphyix documentation and pushes it online'
	@echo ' - make git          : push to github repo'
	@echo ' - make pypi         : upload to pypi'
	@echo ' - make release      : all of the above, in order'
	@echo ' - make html         : generate sphyix documentation'
	@echo ' - make examples     : generate sphyix documentation with examples'
	@echo ' - make cocos        : generate list of COCOS transformations'
	@echo ''

tests: tests2 tests3

tests2:
	python2 -m unittest discover --pattern="*.py" -s omas/tests -v

tests3:
	python3 -m unittest discover --pattern="*.py" -s omas/tests -v

tests_plot: tests2_plot tests3_plot

tests2_plot:
	python2 -m unittest discover --pattern="*_plot.py" -s omas/tests -v

tests3_plot:
	python3 -m unittest discover --pattern="*_plot.py" -s omas/tests -v

tests_physics: tests2_physics tests3_physics

tests2_physics:
	python2 -m unittest discover --pattern="*_physics.py" -s omas/tests -v

tests3_physics:
	python3 -m unittest discover --pattern="*_physics.py" -s omas/tests -v

tests_utils: tests2_utils tests3_utils

tests2_utils:
	python2 -m unittest discover --pattern="*_utils.py" -s omas/tests -v

tests3_utils:
	python3 -m unittest discover --pattern="*_utils.py" -s omas/tests -v

requirements:
	rm requirements.txt
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

itm:
	cd omas/utilities && python generate_itm_interface.py

cocos:
	cd omas/utilities && python generate_cocos_signals.py

git:
	git push

pypi:
	python setup.py sdist upload

release: tests2 tests3 requirements json itm docs git pypi
	@echo 'Done!'
