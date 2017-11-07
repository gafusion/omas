# **OMAS** (**O**rdered **M**ultidimensional **A**rray **S**tructure)

OMAS is a set of Python tools that aim at simplifying the interface between third-party codes and the ITER IMAS data storage infrastructure. IMAS is a set of codes, an execution framework, a data model, a data get/put API, and a data storage infrastructure used for manipulating ITER data. The IMAS data model organizes data into a hierarchical tree. There data is stored in the leaf nodes, and the branches are structures or arrays of structures.

The idea behind OMAS is to provide a convenient Python API can store data in a format that is compatible with the IMAS data model, using other storage systems in addition to IMAS itself. The OMAS library provides convenient APIs to convert data between the OMAS and IMAS storage systems.  The ability of OMAS to handle data in an IMAS-compatible way, without relying on the IMAS library itself, exempts codes from such (cumbersome) dependency. Furthermore, any physics code or programming language that is capable of reading/writing data using one of the many OMAS supported data formats (eg. NetCDF) can take advantage of the functionalities provided by OMAS.

OMAS itself does not address the problem of mapping of the physics codes I/O to the IMAS data model. Instead, Python-based integrated modeling frameworks (such as [OMFIT](http://gafusion.github.io/OMFIT-source)) can be used to define wrappers that leverage OMAS to conveniently map the physics codes I/O, and enable their data to be exchanged with IMAS.

The following code snipplet shows the usage of the Python `omas` class:

  ```python
from omas import *
import numpy

# instantiate new omas data structure (ODS)
ods=omas()

# add some 0D data
ods['equilibrium']['time_slice'][0]['time']=1000.
ods['equilibrium']['time_slice'][0]['global_quantities.ip']=1.E6
# add some 1D data
ods['equilibrium']['time_slice'][0]['profiles_1d.psi']=[1,2,3]
# add some 2D data
ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['b_field_tor']=[[1,2,3],[4,5,6]]

# Save to NetCDF file
save_omas(ods,'test.nc')
# Load from NetCDF file
ods1=load_omas('test.nc')
# Check data consistency
if not different_ods(ods, ods1):
   print('OMAS data was saved to and loaded from file correctly')

# Save to IMAS
paths=save_omas_imas(ods, user='meneghini', tokamak='ITER', version='3.10.1', shot=133221, run=0, new=True)
# Load from IMAS
ods1=load_omas_imas(user='meneghini', tokamak='ITER', version='3.10.1', shot=133221, run=0, paths=paths)
# Check data consistency
if not different_ods(ods, ods1):
   print('OMAS data was saved to and loaded from IMAS correctly')
  ```

The hierarchical IMAS data model is represented in OMAS as a set of nested `omas` objects. Techically speaking the Python `omas` class is a subclass of the standard Python dictionary class, and it inherits all of the functionalities, with the addition of:

1. on the fly checks for consistency with IMAS data model

2. graceful error handling with suggestions for navigating the hierarchical data structure

3. dynamic creation of the tree hierarchy as items they are accessed

4. support read/write/access using multiple formats

	* standard Python dictionary format:
	
	  ```python
     ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['psi']
	  ```
   
	* IMAS hierarchical tree format:
   	
	  ```python
	  ods['equilibrium.time_slice[0].profiles_2d[0].psi']
	  ```
	
	* dot separated string format: 
	
	  ```python
	  ods['equilibrium.time_slice.0.profiles_2d.0.psi']
	  ```
	
	* list of nodes format:
	
	  ```python
	  ods[['equilibrium','time_slice',0,'profiles_2d',0,'psi']]
	  ```

5. save/load omas objects to/from:
 
   * Python pickle: files using native Python serialization tools

   * Json: files using Json ASCII format for representing hierarchical data

   * nc: files using binary NetCDF format 

   * S3: the cloud using Amazon Simple Storage Service  

   * IMAS: ITER data storage system

## Installation

* Users typical installation:
  ```python
  sudo pip install omas
  ```
* Developers installation
  ```python
  git clone git@github.com:gafusion/omas.git
  cd omas                                   
  sudo pip install -e .[build_structures]   
  ```
