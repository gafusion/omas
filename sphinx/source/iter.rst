Access data at ITER with OMAS
=============================
1. SSH to the ``hpc-login.iter.org`` server (use the X2GO for interactive/graphical session)

2. Load the IMAS and OMAS unix modules::

       >> module load IMAS OMAS

   .. warning::
     On the ITER clusters the IMAS environment is loaded with ``module load IMAS`` instead of the ``module load imas`` that is commonly used elsewhere.
     The ``module load imas`` will still execute on the ITER clusters, but the environment itself will not work properly!
     For consistency, on the ITER cluster, the OMAS module follows the same convention as the IMAS module.

3. Find what is available in the ITER IMAS database::

       >> scenario_summary

4. Access ITER scenario database via OMAS from ``python``:

   .. code-block:: python

      # load OMAS package
      from omas import *

      # load data from a shot chosen from the ITER scenario database
      ods = load_omas_iter_scenario(shot=131034, run=0)

      # print nodes with data
      from pprint import pprint
      pprint(ods.pretty_paths())

      # save data in different format (eg. pickle file)
      save_omas_pkl(ods, 'iter_scenario_131034.pk')

   For more information on how to manipulate OMAS data see the :ref:`high-level OMAS overview page <concept>`
   and the extensive :ref:`list of OMAS examples <general_examples>`.

Remotely access ITER data with OMAS and OMFIT
=============================================
OMFIT adds remote access capability to the IMAS functions within OMAS (``load_omas_imas_remote``, ``save_omas_imas_remote`` and ``load_omas_iter_scenario_remote`` functions, respectively).
In addition, the ``iter_scenario_summary_remote`` allows querying for the ITER scenario database remotely from within OMFIT.

1. Set the ``MainSettings['SERVER']['ITER_username']`` to your ITER username

2. Remotely query ITER scenario database:

   .. code-block:: python

       OMFIT['iter_scenarios'] = iter_scenario_summary_remote()

3. Access ITER scenario database remotely from within OMFIT:

   .. code-block:: python

       OMFIT['ods'] = load_omas_iter_scenario_remote(shot=131034, run=0)

