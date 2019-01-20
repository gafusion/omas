Accessing IMAS data at ITER with OMAS
=====================================

1. Login to the `hpc-login.iter.org` server (use the X2GO for interactive/graphical session)

2. install OMAS in your ITER account::

       >> pip install --user --upgrade omas

   for more installation instructions read :ref:`the OMAS install page <install>`.

3. Find what is available in the ITER IMAS database
   First one needs to load the IMAS unix module::

       >> module load IMAS

   .. warning::
     On the ITER clusters the IMAS environment is loaded with ``module load IMAS`` instead of the ``module load imas`` that is commonly used elsewhere.
     The ``module load imas`` will still execute on the ITER clusters, but the environment itself will not work properly!

   To browse the ITER scenario database::

       >> pip install --user --upgrade pyyaml # (this needs to be done only once)
       >> scenario_summary

   A list of scenarios will then be printed on screen

4. Access ITER scenario database via OMAS::

       >> python

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
   and the extensive list of :ref:`OMAS examples <general_examples>`.
