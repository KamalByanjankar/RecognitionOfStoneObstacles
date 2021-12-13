import matplotlib.pyplot as plt
import numpy as np

def plot_graphs(data, limit=0):
    limit = limit if limit != 0 else np.array(data).shape[0]
    fig = plt.figure(figsize=(5, 5))
    for index, d in enumerate(data):
        if index >= limit:
            break
#         ax = fig.add_subplot(32, 32, index+1)
        ax = fig.add_subplot(1, 1, index + 1)
        ax.plot(d)        
        
        