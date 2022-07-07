# How to generate these animations

To generate the different animation, the following commands can be used. Thereby, the data used for the initial criminal network can be changed accordingly

    # To simulate the attachment process using the different methods

    # preferential attachment
    python3 main.py -animate-attachment-process -read-data -attach-meth preferential

    # random attachment
    python3 main.py -animate-attachment-process -read-data -attach-meth random

    # small-world attachment
    python3 main.py -animate-attachment-process -read-data -attach-meth preferential


To generate an animation of the simulation, the following command will do it

    python3 main.py -animate-simulation unfiltered -read-data -attach-meth preferential

To have an animation of only the evlution of the criminal network

    python3 main.py -animate-simulation filtered -read-data -attach-meth preferential
