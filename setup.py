from setuptools import setup, find_packages
import subprocess
import pathlib 
import os

here = pathlib.Path(__file__).parent.resolve()

DESCRIPTION = 'SubSahara Solar Estimation'

DIR_NAME=os.path.dirname(os.path.abspath(__file__))

with open(here/"requirements.txt", "r") as f:
    required_packages = f.read().splitlines()

# Setting up
setup(
        name="SuSSE",
        use_scm_version=True,
        setup_requires=['setuptools_scm'],
        description=DESCRIPTION,
        url="https://github.com/Mijan/Irradiation_Estimation",
        package_dir={'': 'src'},
        packages=find_packages(where="./src/"),
        install_requires=required_packages,
        python_requires=">=3.11"
    
)
