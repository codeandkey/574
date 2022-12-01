#!/bin/python

import matplotlib.pyplot as plt
import numpy as np
import sys
from os.path import basename

sources = sys.argv[2:]

colors = ['r', 'g', 'b', 'y']
fig, ax = plt.subplots()

for source in sources:
    data_Y = None

    with open(source, 'r') as f:
        train_Y, test_Y = eval(f.read())

    data_X = np.arange(0, len(train_Y))

    col = colors[0]
    colors = colors[1:]
    colors.append(col)

    ax.plot(data_X,
            train_Y, 
            col + '-',
            alpha=0.4)

    ax.plot(data_X,
            test_Y,
            col + '-', alpha=0.9,
            label=f'test {basename(source)}')

ax.set_title(sys.argv[1])
ax.set_ylabel('accuracy')
ax.set_xlabel('epoch')
ax.legend()

plt.show()
