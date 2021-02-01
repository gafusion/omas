OMAS at ITER
============
.. _iter:

**The ITER IO has installed and is officially supporting a public installation of the OMAS library on the ITER clusters**.

1. SSH to the ``hpc-login.iter.org`` server (use the X2GO for interactive/graphical session)

2. Load the IMAS and OMAS unix modules::

       >> module load IMAS OMAS

Access ITER data
================
Although ITER experimental data is yet to be produced, OMAS can already be used to access the database of ITER plasma scenarios that is curated by the ITER IO.

1. Find what is available in the ITER IMAS database::

       >> scenario_summary

2. Access ITER scenario database via OMAS from ``python``:

   .. code-block:: python

      # load OMAS package
      from omas import *

      # load data from a pulse chosen from the ITER scenario database
      ods = load_omas_iter_scenario(pulse=131034, run=0)

      # print nodes with data
      from pprint import pprint
      pprint(ods.pretty_paths())

      # save data in different format (eg. pickle file)
      save_omas_pkl(ods, 'iter_scenario_131034.pk')

   For more information on how to manipulate OMAS data see the :ref:`high-level OMAS overview page <concept>`
   and the extensive :ref:`list of OMAS examples <general_examples>`.

Remotely access ITER data with OMFIT
====================================
`OMFIT adds remote access capability to the IMAS functions within OMAS <https://omfit.io/code.html#module-classes.omfit_omas>`_

1. Set the ``MainSettings['SERVER']['ITER_username']`` to your ITER username

2. Remotely query ITER scenario database:

   .. code-block:: python

       # get the iter scenario data in OMFIT
       OMFIT['iter_scenarios'] = iter_scenario_summary_remote()

       # filter based on some criterion
       OMFIT['filter'] = OMFIT['iter_scenarios'].filter(
           {'List of IDSs': ['equilibrium',
                             'core_profiles',
                             'core_sources',
                             'summary'],
            'Workflow': 'CORSICA',
            'Fuelling': 'D-T'})

       # sort based on a column
       OMFIT['filter'].sort('Ip[MA]')

       # display filtered and sorted table
       print(OMFIT['filter'])

3. Access ITER scenario database remotely from within OMFIT:

   .. code-block:: python

       OMFIT['ods'] = load_omas_iter_scenario_remote(pulse=131034, run=0)

ITER scenario requirements
==========================
There is :ref:`a subset of IDS fields<ITER scenario requirements>` that are required to add datasets to the ITER scenario database

Tutorial for running predictive ITER modeling with OMFIT+OMAS
=============================================================
* `Google docs <https://docs.google.com/document/d/1g3VStisQ1wIrhn__rkDQ4sBiv7VZcOiLZzbDMvKw1Lg/edit?usp=sharing>`_  or `PDF <https://docs.google.com/document/export?format=pdf&id=1g3VStisQ1wIrhn__rkDQ4sBiv7VZcOiLZzbDMvKw1Lg&token=AC4w5VipgAXUCbfJ2uI9G3tidgRWhSaMFw%3A1554239840631&includes_info_params=true>`_

Tutorial for running the IMAS Python HCD workflow via OMFIT+OMAS
================================================================
* `Google docs <https://docs.google.com/document/d/1ICtPmKgbO10mUvC0HKdeF6BWhg7MIF4sg9QPkTuHuug/edit?usp=sharing>`_  or `PDF <https://docs.google.com/document/export?format=pdf&id=1ICtPmKgbO10mUvC0HKdeF6BWhg7MIF4sg9QPkTuHuug&token=AC4w5VipgAXUCbfJ2uI9G3tidgRWhSaMFw%3A1554239840631&includes_info_params=true>`_
