all:
	@echo 'OMAS makefile help'
	@echo ''
	@echo ' - make docs    : generate sphyix documentation'
	@echo ' - make json    : generate IMAS json structure files'
	@echo ' - make itm     : generate omas_itm.py from omas_imas.py'
	@echo ' - make git     : upload to OMAS github repo'
	@echo ' - make pipy    : upload to pipy'
	@echo ' - make release : all of the above'
	@echo ''

docs:
	cd sphinx && make html && make commit && make push

json:
	cd utilities && python build_json_structures.py

itm:
	cd utilities && python generate_itm_interface.py

git:
	git push

pipy:
	python setup.py sdist upload

release: docs json itm git pipy
	@echo Done!
