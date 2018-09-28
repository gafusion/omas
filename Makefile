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
	@echo ' - make pipy         : upload to pipy'
	@echo ' - make release      : all of the above, in order'
	@echo ' - make html         : generate sphyix documentation'
	@echo ' - make cocos        : generate list of COCOS transformations'
	@echo ''

tests: tests2 tests3

tests2:
	python2 -m unittest discover --pattern="*.py" -s omas/tests/ -v

tests3:
	python3 -m unittest discover --pattern="*.py" -s omas/tests/ -v

requirements:
	rm requirements.txt
	python setup.py --name

html:
	cd sphinx && make html

docs: html
	cd sphinx && make commit && make push

json:
	cd omas/utilities && python build_json_structures.py

itm:
	cd omas/utilities && python generate_itm_interface.py

cocos:
	cd omas/utilities && python generate_cocos_signals.py

git:
	git push

pipy:
	python setup.py sdist upload

release: tests2 tests3 requirements json itm cocos docs git pipy
	@echo 'Done!'
