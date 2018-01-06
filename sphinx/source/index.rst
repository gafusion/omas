Ordered Multidimensional Array Structure
========================================

.. figure:: images/omas_logo_high_res.gif
  :align: center
  :width: 50%
  :alt: OMAS logo
  :target: _images/omas_logo_high_res.gif

OMAS (Ordered Multidimensional Array Structure) is a Python library designed to simplify the interface of third-party codes with the `ITER <http://iter.org>`_ Integrated Modeling and Analysis Suite (IMAS).
ITER IMAS defines a data model, a data get/put API, and a data storage infrastructure used for manipulating ITER data.

At the heart of OMAS is the idea of providing a convenient API which can store data in a format that is compatible with the IMAS data model, but using other storage systems in addition to the one provided by IMAS itself.
Within OMAS data compatible with the IMAS data model is easily translated between these different storage systems.
Furthermore, any physics code or programming language that is capable of reading/writing data using one of the many OMAS supported data formats (eg. NetCDF) can take advantage of the functionalities provided by OMAS.

OMAS itself does not address the problem of mapping of the physics codes I/O to the IMAS data model.
Such mappings are defined in third party Python codes and frameworks, as done for example with the data classes of the `OMFIT framework <http://gafusion.github.io/OMFIT-source>`_.
