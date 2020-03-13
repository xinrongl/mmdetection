import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def crop(image, return_array=False, shape=(3, 3), plot=True, figsize=(10, 10)):
    """
    Slice image into multiple images.
    Params:    
        image: str/pathlib object, image directory
        return_array: bool, if true, return sliced images as np.array
        plot: bool, if true, plot sliced images
        shape: tuple (nrows, ncols), number of images the raw input will be sliced into
        figsize: tuple
    Return: list of np.array contained sliced images
    """
    im_np = np.array(Image.open(image))
    M = im_np.shape[0]//shape[0]
    N = im_np.shape[1]//shape[1]
    
    tiles = [im_np[x:x+M,y:y+N] for x in range(0,im_np.shape[0],M) for y in range(0,im_np.shape[1],N)]
    if plot:
        fig, axs = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=figsize)
        axs = axs.flatten()
        for tile, ax in zip(tiles, axs):
            ax.imshow(tile)
        plt.show()
    if return_array:
        return tiles
    return None



def reconstruct(images, return_array=False, shape=(3, 3), plot=True, figsize=(10, 10)):
    """
    Reversed outputs from crop and combined with to a single image. 
    """
    img = np.vstack([np.column_stack(tiles[i: i+shape[1]]) for i in range(0, len(images), shape[1])])
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        plt.show()
    if return_array:
        return img
    return None
