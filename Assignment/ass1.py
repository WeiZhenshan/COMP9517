import numpy as np
import cv2
import sys
from copy import deepcopy

def max_filter(I,N):
    A = deepcopy(I)
    for i in range(0,len(I)):
        for j in range(0,len(I[0])):
            r1=i-N
            r2=i+N
            s1=j-N
            s2=j+N
            if r1<0:
                r1=0

            if r2>len(I)-1:
                r2=len(I)-1

            if s1<0:
                s1=0

            if s2>len(I[0])-1:
                s2=len(I[0])-1

            max=0
            for t in range(r1,r2):
                for u in range(s1,s2):
                    if I[t][u]>max:
                        max=I[t][u]

            A[i][j]=max
    return A

def min_filter(A,N):
    B = deepcopy(A)
    for i in range(0,len(A)):
        for j in range(0,len(A[0])):
            r1=i-N
            r2=i+N
            s1=j-N
            s2=j+N
            if r1<0:
                r1=0

            if r2>len(A)-1:
                r2=len(A)-1

            if s1<0:
                s1=0

            if s2>len(A[0])-1:
                s2=len(A[0])-1

            min=255
            for t in range(r1,r2):
                for u in range(s1,s2):
                    if A[t][u]<min:
                        min=A[t][u]

            B[i][j]=min
    return B

def subtract_p(I,B):
    O = deepcopy(I)
    O = O.astype('int32')
    for i in range(0,len(I)):
        for j in range(0,len(I[0])):
            O[i][j]=int(I[i][j])-int(B[i][j])+255
    return O

def subtract_c(I,B):
    O = deepcopy(I)
    O = O.astype('int32')
    for i in range(0, len(I)):
        for j in range(0, len(I[0])):
            O[i][j] = int(I[i][j]) - int(B[i][j])
    return O

def normalize(O):
    cv2.normalize(src=O,dst=O,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
    return O



I=cv2.imread('Particles.png',0) # 0 for gray-scale
I2=cv2.imread('Cells.png',0)
print('input N')
n=int(input())
N=n//2
print('input M')
M=int(input())
if M==0:
    I = cv2.imread('Particles.png', 0)
    A=max_filter(I,N)
    # cv2.imwrite('particles_max_filter9.png', A)
    B=min_filter(A,N)
    # cv2.imwrite('particles_min_filter9.png', B)
    O=subtract_p(I,B)
    # cv2.imwrite('particles_sub_9.png',O)
    O=normalize(O)
    cv2.imwrite('particles_out.png',O)

if M==1:
    I = cv2.imread('Cells.png', 0)
    A = min_filter(I, N)
    # cv2.imwrite('cells_min_filter37.png',A)
    B = max_filter(A, N)
    # cv2.imwrite('cells_max_filter37.png', B)
    O = subtract_c(I, B)
    # cv2.imwrite('cells_sub_37.png', O)
    O = normalize(O)
    cv2.imwrite('cells_out.png', O)


