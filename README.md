# Criminal_network_analysis
Analysing the resilience of criminal networks in an iterative fashion.


---
## Setup
### Requirements
* Ubuntu 20.04 / MacOS X
* Python 3.8

### Environment
In this repo two environments are used, Pipenv and conda. Pipenv is a light tool, which makes sure that the versions and dependencies are correct. However, to install tools that build on other languages such as C/C++, for mac, conda is preferred. Conda has its own complier which makes it really easy to install packages

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

    # First install all missing dependencies
    # In my case:

    $ sudo apt-get install libcgal-dev
    $ sudo apt-get install expat
    $ sudo apt install libsparsehash-dev

    # Activate python environment
    $ git clone https://github.com/pygobject/pycairo.git
    $ cd pycairo/
    $ python3 setup.py build
    $ python3 setup.py install

    $ cd graph-tool-X.XX
    $ ./configure --prefix=$HOME/.local
    $ make install


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
