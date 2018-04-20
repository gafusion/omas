all:
	@echo 'OMAS makefile help'
	@echo ''
	@echo ' - make json    : generate IMAS json structure files'
	@echo ' - make docs    : generate sphyix documentation'
	@echo ' - make itm     : generate omas_itm.py from omas_imas.py'
	@echo ' - make release : push new OMAS release to pipy'
	@echo ''

docs:
	cd sphinx && make html && make commit && make push

json:
	cd utilities && python build_json_structures.py

itm:
	cd utilities && python generate_itm_interface.py

release: docs json itm
	python setup.py sdist upload
