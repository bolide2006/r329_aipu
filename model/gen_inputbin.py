from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

input_height=224
input_width=224
input_channel = 3
mean = [127.5, 127.5, 127.5]
var = 1

img_name = "1.jpg"
img = Image.open(img_name)
plt.imshow(img)

img_w, img_h = img.size
if img_w/img_h > input_width/input_height :
    tmp_h = input_height
    tmp_w = int(input_height/img_h*img_w)
    oft_y = 0
    oft_x = (tmp_w-input_width)/2
else:
    tmp_w = input_width
    tmp_h = int(input_width/img_w*img_h)
    oft_x = 0
    oft_y = (tmp_h-input_height)/2
img1 = img.resize((tmp_w, tmp_h),Image.ANTIALIAS)
plt.imshow(img1)
img2 = img1.crop((oft_x,oft_y,oft_x+input_width,oft_y+input_height))
plt.imshow(img2)

img_arr = (np.array(img2)-mean)/var
img_arr=img_arr.astype(np.int8)

# 保存成仿真需要的bin文件
import struct
data=b''
for y in range(img_arr.shape[1]):
    for x in range(img_arr.shape[0]):
        data += struct.pack('bbb',img_arr[y,x,0],img_arr[y,x,1],img_arr[y,x,2]) 
        
fw = open("input.bin", "wb")
fw.write(data)
fw.close()
print("save to input.bin OK")