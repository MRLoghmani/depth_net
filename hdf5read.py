import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

f = h5py.File(sys.argv[1],'r')

ks = f.keys()

for index,key in ks:
    print "these are the index and the key:"
    print index, key
    print "this is the data:"
   # import code
   # code.interact(local=locals())
    print "this is the f[index]:"
    print f[index]
    print "this is the f[key]:"
    print f[key]
   
   # data = np.array(f[key].value())
    exit()

