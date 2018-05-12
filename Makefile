all:
	@echo 'OMAS makefile help'
	@echo ''
	@echo ' - make docs         : generate sphyix documentation'
	@echo ' - make json         : generate IMAS json structure files'
	@echo ' - make itm          : generate omas_itm.py from omas_imas.py'
	@echo ' - make git          : push to github repo'
	@echo ' - make pipy         : upload to pipy'
	@echo ' - make hard_deps    : copy OMAS to directories specified by $OMAS_HARD_DEPS'
	@echo ' - make release      : all of the above, in order'
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

OMAS_HARD_DEPS ?= None
LAST_DIST=`ls -tr dist | tail -1`
ifneq (${OMAS_HARD_DEPS},None)
hard_deps:
	python setup.py sdist 
	echo ${OMAS_HARD_DEPS} | tr ':' ' ' | xargs -n 1 tar -xvf dist/${LAST_DIST} --strip 1 -C
else
hard_deps:
	@echo '$OMAS_HARD_DEPS is not defined'
endif

release: docs json itm git pipy hard_deps
	@echo 'Done!'
