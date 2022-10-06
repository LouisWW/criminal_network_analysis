# Longitudinal analysis of the topology of criminal networks using a simple cost-benefit agent-based model



<p align="center">

![Works with Ubuntu](https://img.shields.io/badge/Ubuntu-v20.04--LTS-blue?style=flat-square)
![Tested with](https://img.shields.io/badge/Pytest-80%25%20coverage-red)
![Python](https://img.shields.io/badge/python-v3.8-green)

</p>

---

In the context of the master thesis for the [Computational Science](https://www.uva.nl/en/programmes/masters/computational-science/computational-science.html) programme at the [University of Amsterdam](https://www.uva.nl/en) and the [Vrije Universiteit Amsterdam](https://vu.nl/nl),  the following repository contains all the code to reproduce the simulations and the results presented in the thesis. The subject of the thesis is the understanding of the dynamics of a criminal network within a population. Thus, based on the following [article](https://pubmed.ncbi.nlm.nih.gov/31278354/), a cost-benefit model was further developed by implementing an explicit network structure. In other words, the interactions between agents are dictated by their social ties. In this way, the evolution of the criminal network can be simulated and analysed.  Thus, characteristics such as density, information flow or size of the largest component are analysed. The figure below visualise the simulation of a social network, where red nodes correspond to members of a criminal organisation, blue nodes correspond to independent criminals and green nodes represent law-abiding civilians.


![Alt Text](src/results/video/simulation_preferential.gif)

Based on the population shown above, the members of a criminal organisation are filtered out, generating the following criminal network. The evolution of this specific network is analysed using social network analysis.

![Alt Text](src/results/video/simulation_filtered_preferential.gif)

---

## Code

Folder [src](/src) contains all  the code for this project.

---

## Setup
### Requirements
* Ubuntu 20.04 / MacOS X
* Python 3.8

### Environment
In this repo two environments are used, Pipenv and conda. Pipenv is a light tool, which makes sure that the versions and dependencies are correct. However, to install tools that build on other languages such as C/C++, for mac, conda is preferred. Conda has its compiler which makes it easy to install packages. **It is possible that some packages are missing in Pipfile/conda_environment.yml and need to be manually added with pipenv install.../pip3 install... respectively!!!**

### Ubuntu 20.04
#### Pipenv (preferred for Ubuntu)
To setup pipenv environment, python 3.8 needs to be installed first. Then the following lines will install the environment

    $ python3.8 -m pip install pip --upgrade
    $ python3.8 -m pip install pipenv
    $ pipenv install --python 3.8       # create env at root of the directory
    $ pipenv shell                      # to activate env
    $ pipenv install                    # to get all the dependencies

Additionally to the pipenv environment, a python package called **graph-tool** is used. The installation doesn't work through pipenv installing system. To install the package, the manual compilation was used. First the package was downloaded from https://graph-tool.skewed.de/

    # Install Graph-tool

    # Make sure to activate python environment
    # First install all missing dependencies
    # In my case:

    $ sudo apt-get install libboost-all-dev
    $ sudo apt-get install libcgal-dev
    $ sudo apt-get install expat
    $ sudo apt install libsparsehash-dev
    $ sudo apt install libcairomm-1.0-dev

    $ git clone https://github.com/pygobject/pycairo.git
    $ cd pycairo/
    $ python3 setup.py build
    $ python3 setup.py install

    $ cd graph-tool-X.XX
    $ ./configure --prefix=$HOME/.local
    $ make install

To profile the code, the library pycallgraph is used. The package depends on graphviz, which has to be installed via the ubunut installing system

    $ sudo apt install graphviz

### Mac OSX
##### Conda (preferred for Mac OSX)
To install the conda environment, make sure to install conda fisrt via https://www.anaconda.com/products/individual. Then to create the environment, use the following commands,

    conda env create -f conda_environment.yml

To activate the environment

    conda activate criminal_env

To update the environment

    conda env update --file conda_environment.yml

#### Trouble shooting with clang version

If you try to run some code outside of conda and you experience some troubles with gcc or clang complier, do the following

    xcode-select --install
    sudo xcode-select --switch /Library/Developer/CommandLineTools
    clang --version

    # add in ~/.bashrc_profile
    export CC=/Library/Developer/CommandLineTools/usr/bin/clang
    export CXX=/Library/Developer/CommandLineTools/usr/bin/clang++
    export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)


---
#### Pre-commit

A pre-commit hook is used in this repo to uniform the linting. Thus after the first commit, some additional packages will automatically be installed. Commit will only go through if the pylinting is successful! More info can be found on https://www.youtube.com/watch?v=psjz6rwzMdk&ab_channel=mCoding

    # To initialized pre-commit hooks
    # Make sure to be in the pipenv env or conda env

    $ pre-commit install

When using pre-commit for the first time, it will download all the packages (flake8,blake, pydocstring). So don't be afraid!!
To make it work correctly you need to do following commands

    git add CHANGED_FILES
    git commit -m"Your message"

Pre-commit is running. If you don't have [Branch_name commit_number] in the git commit output, the pre-commit hook did some changes to it or didn't accept your changes.
What you need to do it either change your stuff, if an error is shown and/or do again until commit is accepted.

    git add CHANGED_FILES
    git commit -m"Your message"

---
#### Compile cython

The last step to run the code is to compile the cython code (.pyx files)
To do so, run in [src](/src) the following command

    make compile_cython
