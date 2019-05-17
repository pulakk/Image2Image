import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def ImageGenerator(path_x, batch_size, shape):
    im_size = shape[:-1]
    channels = shape[-1]

    X = []
    while True:
        for file in os.listdir(path_x):
            x = cv2.imread(os.path.join(path_x, file))
            x = cv2.resize(x, im_size)
            if channels == 1:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                x = np.reshape(x, x.shape+(1,))
            else:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


            X.append(x)

            if len(X)>=batch_size:
                yield np.array(X)/255.0
                X = []
            

def flatten_gray(image):
    return np.reshape(image, image.shape[:-1])

def plot_multiple(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = plt.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    channels = samples.shape[-1]

    for i in range(samples.shape[0]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channels == 1:
            plt.imshow(flatten_gray(samples[i]), cmap='Greys_r')
        else:
            plt.imshow(samples[i])

    return fig