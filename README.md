# criminal_network_analysis
Analysing the resilience of criminal networks in an iterative fashion.


---
## Setup
### Requirements
* Ubuntu 20.04 / MacOS X
* Python 3.8

To setup pipenv environment, python 3.8 needs to be installed first. Then the following lines will install the environment

    $ python3.8 -m pip install pip --upgrade
    $ python3.8 -m pip install pipenv
    $ pipenv install --python 3.8       # create env at root of the directory
    $ pipenv shell                      # to activate env
    $ pipenv install                    # to get all the dependencies

A pre-commit hook is used in this repo to uniform the linting. Thus after the first commit, some additional packages will automatically be installed. Commit will only go through if the pylinting is successful! More info can be found on https://www.youtube.com/watch?v=psjz6rwzMdk&ab_channel=mCoding

    # To initialized pre-commit hooks
    # Make sure to be in the pipenv env

    $ pre-commit install

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



---
