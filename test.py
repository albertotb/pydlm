#!/usr/bin/env python
import sys
import base64
import pickle
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pympler import asizeof
from pydlm import dlm, odlm, trend

ts = [
 0.5429682543922109,
 0.5296058346035057,
 0.5403294585554494,
 0.542441925561093,
 0.5435209708555084,
 0.5430676782288945,
 0.5429877208796179,
 0.5429721282202071,
 0.5429690254184671,
 0.5449758960859548,
 0.5457612294317765,
 0.5434065016617284,
 0.5430519745276086,
 0.5436459000038072,
 0.5437794184525637]

def print_size(obj, level=0):
    print(''.join(['  '] * level) + str(obj))
    for el in obj.refs:
        print_size(el, level=level+1)

def compare_size(obj1, obj2, level=0):
    print(''.join(['  '] * level) + str(obj1))
    print(''.join(['  '] * level) + str(obj2))
    for el1, el2 in zip(obj1.refs, obj2.refs):
        compare_size(el1, el2, level=level+1)

model1 = odlm([]) + trend(degree=2, discount=0.95, name='trend1')
model1.stableMode(False)

model2 = dlm([]) + trend(degree=2, discount=0.95, name='trend1')
model2.stableMode(False)

d1 = {}
d2 = {}
for idx, el in enumerate(ts):
    model1.append([el], component='main')
    model1.fitForwardFilter()

    model2.append([el], component='main')
    model2.fitForwardFilter()

    a1 = asizeof.asized(model1, detail=4)
    a2 = asizeof.asized(model2, detail=4)

    mean1, var1 = model1.predictN(N=1, date=model1.n-1)
    mean2, var2 = model2.predictN(N=1, date=model2.n-1)

    np.testing.assert_almost_equal(mean1, mean2, decimal=7, err_msg='', verbose=True)
    np.testing.assert_almost_equal(var1, var2, decimal=7, err_msg='', verbose=True)

    s1 = base64.b64encode(pickle.dumps(model1)).decode('utf-8')
    s2 = base64.b64encode(pickle.dumps(model2)).decode('utf-8')

    if idx > 0:
        compare_size(a, a1)
    a = a1

    d1[idx] = (a1.size, len(s1))
    d2[idx] = (a2.size, len(s2))
    print(''.join(['-']*80))


df1 = pd.DataFrame.from_dict(d1, orient="index")
df2 = pd.DataFrame.from_dict(d2, orient="index")

## Vemos los resultados
print(df1)
print(df2)
fig, ax = plt.subplots()
df1.join(df2, lsuffix='odlm', rsuffix='dlm').plot(ax=ax)
fig.savefig("memory.pdf")
