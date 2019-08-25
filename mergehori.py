from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import easygui
font1 = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", 65)

#overlap = 100
overlap = int(input('overlap = '))
name1 = easygui.fileopenbox()
name2 = easygui.fileopenbox()
img1 = Image.open(name1)
img2 = Image.open(name2)
width1, height1 = img1.size
width2, height2 = img2.size
if height1 == height2:
    toImage = Image.new('RGBA',(width1 + width2 - overlap, height1))
    draw = ImageDraw.Draw(toImage)
    toImage.paste(img2, (width1 - overlap, 0))
    toImage.paste(img1, (0, 0))
    toImage.save('merged_hori.png')
else:
    print('heights not equal!')

