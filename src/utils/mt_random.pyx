# distutils: sources=["mt19937ar.c"]
cimport utils.mt

def init_state(unsigned long s):
    utils.mt.init_genrand(s)

def random_c():
    return utils.mt.genrand_real3()
