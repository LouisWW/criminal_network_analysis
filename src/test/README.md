# Testing Framework

The Test folder regroups all the testing function. A test function compares the outcome of a given function to the a known value. If the outcome is not equal to the known value, the function being tested is not valid. It does not work as expected. Thus it is important to test
critical function in order to be sure that they return the desired values.

For the testing frame pytest is chosen which is build on the build-in module unittest.


---
## Requirements

    pytest = "*"
    pytest-xdist = "*"
    hypothesis = "*"
    pytest-subtests = "*"
    pytest-cov = "*"

## Commands
The tests are run from the **src** directory with the following command

    # linux
    python3 -m pytest  && find . -name '.coverage.*' -exec rm {} \;

    # mac
    python3 -m pytest  && rm .coverage*

    (command after && makes sure the annoying files are deleted)



This command will run pytest, which is defined in **src/pytest.ini**.
In src/pytest.ini we have:

 -the variable *marker* defines the test sets.

 -the variable *filterwarnings* suppress specific warnings

 -the variable *addopt* defines how/which tests are run:

    Flags for addopt:
        -v                              verbose
        -n                              number of cores:
                                            maximum number of cores-1
                                            otherwise
                                            pytest fails

        -m                              defines which test sets to run:


        -k                              run specific modules(python files)/tests:

        --cov="."                      coverage of all the files in .
        --cov-report html               generate html file
        --cov-config=test/.coveragerc   specify folder where files are generated


    Examples:

    # Run all the tests without generating a coverage report (faster)
    addopts = -v -n 2

    # Run all the local tests
    addopts = -v -n 2 -m "local"

    # Run one specific test
    addopts = -v -k "specific_test_func"

    # Run all the tests form a specific module
    addopts = -v -k "TestModule"

    # Run all the tests with a coverage report
    addopts = -v -n 2 --cov="." --cov-report html --cov-config=test/.coveragerc

## Structure
For pytest (and also unittest), the tests need to be in a file starting with **test_\*.py**. For this repository, all the tests files are contained in the folder called **test**.
To have a clearer structure, subfolders (**tests_\***)are used, which represent the actual folder. Accordingly, each **test_\*.py** represents an actual python file. Please note that pytest.ini is not in the test folder (unfortunately), because otherwise, pytest can't find it...

### Overview

    src
    ├─ pytest.ini                              (define how to run tests)
    │
    └─test
    ├─ README.md
    ├─ conftest.py                          (pytest fixtures a.k.a input usable for other tests)
    │
    ├─ cov_report
    │  └─ index.html                        (open with chrome for overview)


## Conftest
In the **conftest.py** file, so-called fixtures can be created. The idea behind fixtures is that they define a piece of code that can run once (even if called multiple times). Thus this feature allows for an intense snippet of codes to run only once and be used multiple times.
The usage can be defined by a decorator:

    # runs once for the whole testing procedure
    @pytest.fixture(scope='session')
    # runs every time a module is using that feature
    @pytest.fixture(scope='module')

In our case, the conftest.py should contain mock DataFrames or mock classes that can be used for multiple tests.

## Coverage Report

The pytest-cov package is used to create a coverage report. This report can be found in the cov_report folder. Open src/htmlcov/index.html in your browser file with Chrome and not Safari!
