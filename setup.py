from setuptools import setup
import glob,os

#add .json IMAS structure files to the package
here=os.path.abspath(os.path.split(__file__)[0])+os.sep

setup(
    name='omas',
    version='0.1.1',
    description='Ordered Multidimensional Array Structure',
    url='https://gafusion.github.io/omas',
    author='Orso Meneghini',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='integrated modeling OMFIT IMAS ITER',
    packages=['omas','omas.imas_structures.3_10_1'],
    package_data={
        'omas.imas_structures.3_10_1': ['*.json'],
    },
    install_requires=['numpy','netCDF4','boto3'],
    extras_require={
        'imas': ['imas'],
        'build_structures': ['pandas','xlrd']
    }
)
