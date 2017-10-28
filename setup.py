from setuptools import setup
import glob,os

#add .json IMAS structure files to the package
here=os.path.abspath(os.path.split(__file__)[0])+os.sep
if not os.path.exists(here+'MANIFEST.in'):
    with open(here+os.sep+'MANIFEST.in','w') as f:
        for item in glob.glob(here+'omas'+os.sep+'imas_structures'+os.sep+'*'+os.sep+'*.json'):
            f.write('include '+os.path.abspath(item)[len(here):]+'\n')

setup(
    name='omas',
    version='0.0.5',
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
    packages=['omas'],
    install_requires=['numpy','netCDF4','boto3'],
    extras_require={
        'iams': ['imas'],
        'build_structures': ['pandas','xlrd']
    }
)