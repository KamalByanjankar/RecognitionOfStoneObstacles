import matplotlib.pyplot as plt
import numpy as np
import math
def plot_graphs(data, limit=0, length=25):
    size = math.ceil(math.sqrt(length))
    limit = limit if limit != 0 else np.array(data).shape[0]
    fig = plt.figure(figsize=(20, 20))
    for index, d in enumerate(data):
        if index >= limit:
            break
        ax = fig.add_subplot(size, size, index+1)
#         ax = fig.add_subplot(2, 2, index + 1)
        ax.plot(d)        
        
def plot_abs_graphs(data):
    fig = plt.figure(figsize=(40, 30))
    for index, d in enumerate(data):
        ax = fig.add_subplot(8, 8, index+1)
        ax.plot(np.abs(d[4000:]))
        
def plot_scale_graphs(data):
    fig = plt.figure(figsize=(40, 30))
    for index, d in enumerate(data):
        ax = fig.add_subplot(8, 8, index+1)
        ax.plot(np.array(d[4000:]) * 2)
        
        