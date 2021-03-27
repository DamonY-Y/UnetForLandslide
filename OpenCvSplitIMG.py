import cv2
import os

img = cv2.imread('Image.png')  # 写image路径，导入

m = img.shape[0]
n = img.shape[1]

label = cv2.imread('Label.png')  # 写label路径，导入
x = label.shape[0]
y = label.shape[1]

print(m, n)
print(x, y)

pic_size = 81  # 切割的图片大小为 80*80
a = m//pic_size  # 切割的横向个数
b = n//pic_size   # 切割的纵向个数

img_path = './data/img_split'
label_path = './data/label_split'

if not os.path.exists(img_path):
    os.mkdir(img_path)
if not os.path.exists(label_path):
    os.mkdir(label_path)

name = ''  # 用作图片的名称
for i in range(a):
    row = i*pic_size
    first_two = str(i) if i > 9 else '0'+str(i)

    for j in range(b):
        col = j * pic_size

        last_two = str(j) if j > 9 else '0' + str(j)
        name = first_two + last_two

        split_imgpth = img_path+r'/'+name+'.png'
        split_labelpth = label_path+r'/'+name+'.png'

        split_img = img[row:row+pic_size-1, col:col+pic_size-1]
        split_label = label[row:row+pic_size-1, col:col+pic_size-1]

        cv2.imwrite(split_imgpth, split_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(split_labelpth, split_label, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(name+'finished!')

