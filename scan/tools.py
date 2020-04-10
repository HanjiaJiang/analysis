'''
File and image handling tools
'''
import os
from PIL import Image
from tkinter import Tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt


def openfile():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = filedialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
    # filename = filedialog.askopenfilename(initialdir = "~/Documents/") # show an "Open" dialog box and return the path to the selected file
    return filename


def selectfolder():
    Tk().withdraw()
    foldername = filedialog.askdirectory()
    return foldername


def hori_join(names, done_name):
    # initial
    w_sum = 0
    img = Image.open(names[0])
    h = img.size[1]

    # get widths
    for i, name in enumerate(names):
        img = Image.open(name)
        w = img.size[0]
        w_sum += w

    # paste
    toImage = Image.new('RGBA', (w_sum, h))
    w_tmp = 0
    for i, name in enumerate(names):
        img = Image.open(name)
        toImage.paste(img, (w_tmp, 0))
        w_tmp += img.size[0]

    toImage.save(done_name + '.png')


def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    # plt.show()
    plt.savefig('combined.png')


if __name__ == "__main__":
    cache = {}
    for file in os.listdir():
        if file.endswith('.png'):
            h = file.split('_')[0]
            if h not in cache:
                cache[h] = [file]
            else:
                cache[h].append(file)
    for k in cache.keys():
        hori_join(sorted(cache[k]), k)
