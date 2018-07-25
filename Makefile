all:
	@echo 'OMAS makefile help'
	@echo ''
	@echo ' - make tests        : run all regression tests'
	@echo ' - make requirements : build requirements.txt'
	@echo ' - make docs         : generate sphyix documentation and pushes it online'
	@echo ' - make json         : generate IMAS json structure files'
	@echo ' - make itm          : generate omas_itm.py from omas_imas.py'
	@echo ' - make git          : push to github repo'
	@echo ' - make pipy         : upload to pipy'
	@echo ' - make release      : all of the above, in order'
	@echo ' - make html         : generate sphyix documentation'
	@echo ''

tests:
	python -m unittest discover --pattern="*.py" -s omas/tests/

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

git:
	git push

pipy:
	python setup.py sdist upload

release: requirements json docs itm git pipy
	@echo 'Done!'
