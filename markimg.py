from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import easygui
#font1 = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", 65)

img1 = Image.open(easygui.fileopenbox())

toImage = Image.new('RGBA',img1.size)
draw = ImageDraw.Draw(toImage)
toImage.paste(img1, (0, 0))
width, height = img1.size
border = 20
interval = 5.6

# font
font1 = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 60)
font2 = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 50)

# labels
draw.text((width*2.2/10.0, border),'(A) firing rate',(0,0,0), font=font1)
draw.text((width*6.6/10.0, border),'(B) pairwise correlation',(0,0,0), font=font1)
draw.text((5, height/6.2),'L2/3',(0,0,0), font=font2)
draw.text((5, height/6.2 +height/interval),'L4',(0,0,0), font=font2)
draw.text((5, height/6.2 +2*height/interval),'L5',(0,0,0), font=font2)
draw.text((5, height/6.2 +3*height/interval),'L6',(0,0,0), font=font2)

toImage.save('marked.png')

