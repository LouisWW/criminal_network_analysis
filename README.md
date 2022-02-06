# criminal_network_analysis
Analysing the resilience of criminal networks in an iterative fashion.


---
## Setup
### Requirements
* Python 3.8
* Pipenv Virtual Environment

#### To setup pipenv environment, python 3.8 needs to be installed first. Then the following lines will install the environment
    python3.8 -m pip install pip --upgrade
    python3.8 -m pip install pipenv
    pipenv install --python 3.8  # create env at root of directory
    pipenv shell                 # to activate env
    pipenv install               # to get all the dependencies

#### A pre-commit hook is used in this repo to uniform the linting. Thus after the first commit, some additional packages will automatically be installed. Commit will only go through if the pylinting is successful!
---
