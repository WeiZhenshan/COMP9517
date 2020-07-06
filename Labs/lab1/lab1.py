import cv2
import matplotlib.pyplot as plt
import numpy as np
###########ques1########################
def contrast_stretch(img):

    a=0
    b=255
    c=img.min()
    d=img.max()
    # print(c)
    # print(d)

    ratio=(b-a)/(d-c)
    out=img.copy()

    for i in range(0,len(img)):
        for j in range(0,len(img[0])):
            # for k in range(0,len(img[0][0])):

            out[i][j]=(img[i][j]-c)*ratio+a
    # print(out.min())
    # print(out.max())
    return out

########################################

#######ques2############################
# histogram=cv2.calcHist([img],[0],None,[256],[0,256])
# plt.plot(histogram)
# plt.show()
#######################################


# gray = cv2.imread('cat.png',0)
# grad_x=cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
# grad_y=cv2.Sobel(gray,cv2.CV_8U,0,1,ksize=3)
# cv2.imwrite('sobelx.png',grad_x)
# cv2.imwrite('sobely.png',grad_y)

########## ques 4 ########################

I=cv2.imread('cat.png',0)
L=cv2.GaussianBlur(src=I,ksize=(0,0),sigmaX=3.0)    #blur
H=cv2.subtract(I,L)         #subtract
a=1.25
H_=a * H                    #multiply with a constant
I=np.float64(I)              #change data type
res=cv2.add(I,H_)                   #add
res=np.uint8(res)           #change data type
O=contrast_stretch(res)     #contrast stretching
cv2.imwrite('O.png', O)     #save resulting image file

###########################################################
# cv2.imwrite('blur0.png',L)

# cv2.imwrite('H.png',H)
# print(H.dtype)


#print(H_.dtype)
# print(res.max())
# print(res.min())

# I=np.int32(I)
# L=np.int32(L)
# H=np.subtract(I,L)
# H_=np.uint8(H)
# H_=cv2.normalize(H,H_,0,255,cv2.NORM_MINMAX)
# print(H.dtype)
# cv2.imwrite('H.png',H)