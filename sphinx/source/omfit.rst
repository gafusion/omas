OMAS provides the backbone infrastructure that enables centralized data exchange among different physics components
within the `OMFIT <https://omfit.io>`_ framework, as embodied in the
`STEP module <https://omfit.io/modules/mod_STEP.html#step>`_.

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

* `Remotely query ITER scenario database <https://omfit.io/code.html#classes.omfit_omas.iter_scenario_summary_remote>`_:

  .. code-block:: python

     OMFIT['iter_scenarios'] = iter_scenario_summary_remote()

* `Access ITER scenario database remotely from within OMFIT <https://omfit.io/code.html#classes.omfit_omas.load_omas_iter_scenario_remote>`_:

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

* OMFITcotsim

  *  `OMFITcotsim.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_cotsim.OMFITcotsim.to_omas>`_
* OMFITdprobe

  *  `OMFITdprobe.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_efund.OMFITdprobe.to_omas>`_
* OMFITfbm

  *  `OMFITfbm.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_transp.OMFITfbm.to_omas>`_
* OMFITgenray

  *  `OMFITgenray.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_genray.OMFITgenray.to_omas>`_
* OMFITgeqdsk

  *  `OMFITgeqdsk.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_eqdsk.OMFITgeqdsk.to_omas>`_
  *  `OMFITgeqdsk.from_omas() <https://omfit.io/code.html#omfit_classes.omfit_eqdsk.OMFITgeqdsk.from_omas>`_
* OMFITinputgacode

  *  `OMFITinputgacode.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_gapy.OMFITinputgacode.to_omas>`_
  *  `OMFITinputgacode.from_omas() <https://omfit.io/code.html#omfit_classes.omfit_gapy.OMFITinputgacode.from_omas>`_
* OMFITinputprofiles

  *  `OMFITinputprofiles.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_gapy.OMFITinputprofiles.to_omas>`_
  *  `OMFITinputprofiles.from_omas() <https://omfit.io/code.html#omfit_classes.omfit_gapy.OMFITinputprofiles.from_omas>`_
* OMFITkeqdsk

  *  `OMFITkeqdsk.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_eqdsk.OMFITkeqdsk.to_omas>`_
  *  `OMFITkeqdsk.from_omas() <https://omfit.io/code.html#omfit_classes.omfit_eqdsk.OMFITkeqdsk.from_omas>`_
* OMFITmeqdsk

  *  `OMFITmeqdsk.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_eqdsk.OMFITmeqdsk.to_omas>`_
* OMFITmhdin

  *  `OMFITmhdin.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_efund.OMFITmhdin.to_omas>`_
* OMFITpFile

  *  `OMFITpFile.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_osborne.OMFITpFile.to_omas>`_
  *  `OMFITpFile.from_omas() <https://omfit.io/code.html#omfit_classes.omfit_osborne.OMFITpFile.from_omas>`_
* OMFITplasmastate

  *  `OMFITplasmastate.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_transp.OMFITplasmastate.to_omas>`_
* OMFITprofiles

  *  `OMFITprofiles.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_profiles.OMFITprofiles.to_omas>`_
* OMFITstatefile

  *  `OMFITstatefile.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_onetwo.OMFITstatefile.to_omas>`_
* OMFITtranspOutput

  *  `OMFITtranspOutput.to_omas() <https://omfit.io/code.html#omfit_classes.omfit_transp.OMFITtranspOutput.to_omas>`_
* fluxSurfaces

  *  `fluxSurfaces.to_omas() <https://omfit.io/code.html#omfit_classes.fluxSurface.fluxSurfaces.to_omas>`_
  *  `fluxSurfaces.from_omas() <https://omfit.io/code.html#omfit_classes.fluxSurface.fluxSurfaces.from_omas>`_
