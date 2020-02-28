from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def openfile():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(initialdir = "~/Documents/") # show an "Open" dialog box and return the path to the selected file
    return filename


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