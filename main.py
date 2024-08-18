# remaining to optimize code
# to be able to detect handwritten text and able to detect printed text other then this photo

from ultralytics import YOLO
import warnings
import numpy as np
import aspose.words as aw
import cv2
import torch
from PIL import Image
import contour
from docx import Document

document = Document()

model = YOLO('best (1).pt') # find words from whole images


#reading image and model 
img_name = 'guj_pic.png'
img = cv2.imread(img_name)
res = model(img_name)

boxes= res[0].boxes

#converting to black and white
im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,im = cv2.threshold(im,70,255,cv2.THRESH_BINARY)
# print("first shape",im.shape)
#calculating pixel frequancy

col_di = {} 
def count_color(img):
    w,h= img.shape
    for x in range(0,w):
        for y in range(0,h):
            bw = img[x,y]
            if bw in col_di:
                col_di[bw] += 1
            else:
                col_di[bw] = 1

# if background is black inverse to black

count_color(im)
if col_di[0] > col_di[255]:
    im = cv2.bitwise_not(im)

h,w= im.shape

xmin = [] 
ymin = []
xmax = []
ymax = []

#calculating all coordinate of words 
for c,i in enumerate(boxes):
    box = i.xyxy
    box = box[0]
    xmin.append(int(box[0].item()))
    ymin.append(int(box[1].item()))
    xmax.append(int(box[2].item()))
    ymax.append(int(box[3].item()))

ig = []

#putting index of all words which are overlapping
for i in range(len(xmin)):
    if i not in ig:
        uurange = ymin[i] - h*0.025
        # print(h,ymin[i],uurange)
        ulrange = ymin[i] + h*0.025
        lurange = ymax[i] - h*0.025
        llrange = ymax[i] + h*0.025

        x_llrange = xmin[i] - w*0.01
        x_lrrange = xmin[i] + w*0.01
        x_rlrange = xmax[i] - w*0.01
        x_rrrange = xmax[i] + w*0.01
       
        for i1 in range(len(xmin)):
            if i != i1:
                if  ((xmin[i1] > x_llrange and xmin[i1] < x_lrrange) or (xmax[i1] > x_rlrange and xmax[i1] < x_rrrange)) and xmax[i] < xmax[i1] and (ymin[i1] > uurange and ymin[i1]< ulrange) and (ymax[i1] > lurange and ymax[i1] < llrange):
                    ig.append(i)
                    break

x1 = []
x2 = []
y1 = []
y2 = []
# print("ig",ig)

for i in range(len(xmin)):
    if i not in ig:
        # print(i)
        x1.append(xmin[i])
        x2.append(xmax[i])
        y1.append(ymin[i])
        y2.append(ymax[i])

#cropping every words and rectangle around words
h,w = 0,0
co = 0
guj = []

xmin = x1
ymin = y1
xmax = x2
ymax = y2
s = np.argsort(ymin)

ymin = np.array(ymin)
ymax = np.array(ymax)
xmax = np.array(xmax)
xmin = np.array(xmin)


ymin = ymin[s]
ymax = ymax[s]
xmin = xmin[s]
xmax= xmax[s]


avg_di_ymin = 0
for i in range(len(ymin)):
    if i != 0:
        avg_di_ymin = avg_di_ymin + ymin[i] - ymin[i-1]

avg_di_ymin = avg_di_ymin/len(ymin)
breakLine = []

for i in range(len(ymin)):
    if i != 0:
        a = ymin[i] - ymin[i-1]
        if a > (avg_di_ymin*6):
            breakLine.append(i)



x1_new = [[]]
y1_new = [[]]
x2_new = [[]]
y2_new = [[]]

for i in range(len(ymin)):
    if i in breakLine:
        x1_new.append([xmin[i]])
        x2_new.append([xmax[i]])
        y1_new.append([ymin[i]])
        y2_new.append([ymax[i]])
    else:
        x1_new[-1].append(xmin[i])
        y1_new[-1].append(ymin[i])
        x2_new[-1].append(xmax[i])
        y2_new[-1].append(ymax[i])
        

for i in range(len(y1_new)):
    so = np.argsort(x1_new[i])

    x1_new[i] = np.array(x1_new[i])[so]
    y1_new[i] = np.array(y1_new[i])[so]
    x2_new[i] = np.array(x2_new[i])[so]
    y2_new[i] = np.array(y2_new[i])[so]
    
x1l = []
x2l = []
y1l = []
y2l = []

for i in range(len(y1_new)):
    for j in range(len(y1_new[i])):
        x1l.append(x1_new[i][j])
        y1l.append(y1_new[i][j])
        x2l.append(x2_new[i][j])
        y2l.append(y2_new[i][j])

# print(x1l,y1l,x2l,y2l)
for c,i in enumerate(y1l):
    co = co + 1
    x1 = int(x1l[c])
    y1 = int(y1l[c])
    x2 = int(x2l[c])
    y2 = int(y2l[c])
    # # print(x1,y1,x2,y2)
    # im = cv2.rectangle(im,(int(x1l[c]),int(y1l[c])),(int(x2l[c]),int(y2l[c])),(34,23,52),thickness=2)
    # im = cv2.putText(im,str(co),(int(x1l[c]),int(y1l[c])),cv2.FONT_HERSHEY_COMPLEX,1,(23,42,123))
    # # print(x1,x2,y1,y2)
    cr_im = im[int(y1l[c]):int(y2l[c]),int(x1l[c]):int(x2l[c])]

    # print(cr_im)
    wo = contour.sep_single_cha(cr_im)
    
    # print("word:",wo)
    w1 = ''
    for i in wo:
        w1 = w1 + i
    print("word: ",w1)
    guj.append(w1)

final_guj = ''
for i in guj:
    final_guj = final_guj + ' ' +i

print("final",final_guj)

document.add_paragraph(final_guj)

document.save("output.docx")
# cv2.imshow("dsf",im)
# cv2.waitKey(0) 