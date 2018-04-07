# **OMAS** (**O**rdered **M**ultidimensional **A**rray **S**tructure)

OMAS is a set of Python tools that aim at simplifying the interface between third-party codes and the ITER IMAS data storage infrastructure. IMAS is a set of codes, an execution framework, a data model, a data get/put API, and a data storage infrastructure used for manipulating ITER data. The IMAS data model organizes data into a hierarchical tree. There data is stored in the leaf nodes, and the branches are structures or arrays of structures.

The idea behind OMAS is to provide a convenient Python API can store data in a format that is compatible with the IMAS data model, using other storage systems in addition to IMAS itself. The OMAS library provides convenient APIs to convert data between the OMAS and IMAS storage systems.  The ability of OMAS to handle data in an IMAS-compatible way, without relying on the IMAS library itself, exempts codes from such (cumbersome) dependency. Furthermore, any physics code or programming language that is capable of reading/writing data using one of the many OMAS supported data formats (eg. NetCDF) can take advantage of the functionalities provided by OMAS.

OMAS itself does not address the problem of mapping of the physics codes I/O to the IMAS data model. Instead, Python-based integrated modeling frameworks (such as [OMFIT](http://gafusion.github.io/OMFIT-source)) can be used to define wrappers that leverage OMAS to conveniently map the physics codes I/O, and enable their data to be exchanged with IMAS.

For more info, visit: https://gafusion.github.io/omas/scenarios.html
