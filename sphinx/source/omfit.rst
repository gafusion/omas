OMAS provides the backbone infrastructure that enables centralized data exchange among different physics components
within the `OMFIT <http://gafusion.github.io/OMFIT-source>`_ framework, as embodied in the
`STEP module <http://gafusion.github.io/OMFIT-source/modules/mod_STEP.html#step>`_.

Remotely access IMAS data with OMFIT
====================================
OMFIT adds remote access capability to the IMAS functions within OMAS:

* `Remotely load OMAS data to IMAS <https://gafusion.github.io/omas/code/omas.load_omas_imas.html#omas.load_omas_imas>`_

  .. code-block:: python

     ods = load_omas_imas_remote(serverPicker='iter_login', machine='ITER', pulse=123456, run=1)

* `Remotely save OMAS data to IMAS <https://gafusion.github.io/omas/code/omas.save_omas_imas.html#omas.save_omas_imas>`_

  .. code-block:: python

     save_omas_imas_remote(serverPicker='iter_login', ods=ods, machine='DIII-D', pulse=123456, run=1 new=True)

* `Remotely browse available IMAS data <https://gafusion.github.io/omas/code/omas.browse_imas.html#omas-browse-imas>`_

  .. code-block:: python

     dB = browse_imas_remote(serverPicker='iter_login', user=None, pretty=True)

Some ITER specific IMAS utiliities available via OMFIT:

* `Remotely query ITER scenario database <http://gafusion.github.io/OMFIT-source/code.html#classes.omfit_omas.iter_scenario_summary_remote>`_:

  .. code-block:: python

     OMFIT['iter_scenarios'] = iter_scenario_summary_remote()

* `Access ITER scenario database remotely from within OMFIT <http://gafusion.github.io/OMFIT-source/code.html#classes.omfit_omas.load_omas_iter_scenario_remote>`_:

  .. code-block:: python

    OMFIT['ods'] = load_omas_iter_scenario_remote(pulse=131034, run=0)


Translating between legacy formats and IMAS
===========================================

OMFIT provides an effective way to translate between legacy and IMAS via OMAS

.. figure:: images/eq_omas_omfit.png
  :align: center
  :width: 75%
  :alt: OMFIT+OMAS facilitate save/load gEQDSK to/from IMAS
  :target: /.._images/eq_omas_omfit.png

* OMFITgeqdsk

  * `OMFITgeqdsk.to_omas() <http://gafusion.github.io/OMFIT-source/_modules/classes/omfit_eqdsk.html#OMFITgeqdsk.to_omas>`_

  * `OMFITgeqdsk.from_omas() <http://gafusion.github.io/OMFIT-source/_modules/classes/omfit_eqdsk.html#OMFITgeqdsk.from_omas>`_

* OMFITgacode

  * `OMFITgacode.to_omas() <http://gafusion.github.io/OMFIT-source/_modules/classes/omfit_gacode.html#OMFITgacode.to_omas>`_

  * `OMFITgacode.from_omas() <http://gafusion.github.io/OMFIT-source/_modules/classes/omfit_gacode.html#OMFITgacode.from_omas>`_

* FluxSurfaces

  * `FluxSurfaces.to_omas() <http://gafusion.github.io/OMFIT-source/_modules/classes/fluxSurface.html#fluxSurfaces.to_omas>`_

* OMFITplasmastate

  * `OMFITplasmastate.to_omas() <http://gafusion.github.io/OMFIT-source/_modules/classes/omfit_transp.html#OMFITplasmastate.to_omas>`_

* OMFITstatefile

  * `OMFITstatefile.to_omas() <http://gafusion.github.io/OMFIT-source/_modules/classes/omfit_onetwo.html#OMFITstatefile.to_omas>`_

* OMFITosborne

  * `OMFITosborne.to_omas() <http://gafusion.github.io/OMFIT-source/_modules/classes/omfit_osborne.html#OMFITpFile.to_omas>`_
