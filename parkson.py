# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:33:36 2019

@author: kayo_
"""

import glob
import os
import cv2
import numpy as np
from sklearn.svm import SVC

imagem1_g = cv2.imread('images/imagem1.jpg')
imagem2_g = cv2.imread('images/imagem2.jpg')
imagem3_g = cv2.imread('images/imagem3.jpg')
imagem4_g = cv2.imread('images/imagem4.jpg')
imagem5_g = cv2.imread('images/imagem5.jpg')
imagem6_g = cv2.imread('images/imagem6.jpg')
imagem7_g = cv2.imread('images/imagem7.jpg')
imagem8_g = cv2.imread('images/imagem8.jpg')
imagem9_g = cv2.imread('images/imagem9.jpg')
imagem10_g = cv2.imread('images/imagem10.jpg')
imagem11_g = cv2.imread('images/imagem11.jpg')
imagem12_g = cv2.imread('images/imagem12.jpg')
imagem13_g = cv2.imread('images/imagem13.jpg')
imagem14_g = cv2.imread('images/imagem14.jpg')
imagem15_g = cv2.imread('images/imagem15.jpg')


imagem1 = cv2.resize(imagem1_g, (200,200),interpolation = cv2.INTER_AREA)
imagem2 = cv2.resize(imagem2_g, (200,200),interpolation = cv2.INTER_AREA)
imagem3 = cv2.resize(imagem3_g, (200,200),interpolation = cv2.INTER_AREA)
imagem4 = cv2.resize(imagem4_g, (200,200),interpolation = cv2.INTER_AREA)
imagem5 = cv2.resize(imagem5_g, (200,200),interpolation = cv2.INTER_AREA)
imagem6 = cv2.resize(imagem6_g, (200,200),interpolation = cv2.INTER_AREA)
imagem7 = cv2.resize(imagem7_g, (200,200),interpolation = cv2.INTER_AREA)
imagem8 = cv2.resize(imagem8_g, (200,200),interpolation = cv2.INTER_AREA)
imagem9 = cv2.resize(imagem9_g, (200,200),interpolation = cv2.INTER_AREA)
imagem10 = cv2.resize(imagem10_g, (200,200),interpolation = cv2.INTER_AREA)
imagem11 = cv2.resize(imagem11_g, (200,200),interpolation = cv2.INTER_AREA)
imagem12 = cv2.resize(imagem12_g, (200,200),interpolation = cv2.INTER_AREA)
imagem13 = cv2.resize(imagem13_g, (200,200),interpolation = cv2.INTER_AREA)
imagem14 = cv2.resize(imagem14_g, (200,200),interpolation = cv2.INTER_AREA)
imagem15 = cv2.resize(imagem15_g, (200,200),interpolation = cv2.INTER_AREA)

ret,thresh1=cv2.threshold(imagem1,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem1,200,255,cv2.THRESH_BINARY)
imagem1 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem2,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem2,200,255,cv2.THRESH_BINARY)
imagem2 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem3,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem3,200,255,cv2.THRESH_BINARY)
imagem3 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem4,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem4,200,255,cv2.THRESH_BINARY)
imagem4 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem5,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem5,200,255,cv2.THRESH_BINARY)
imagem5 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem6,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem6,200,255,cv2.THRESH_BINARY)
imagem6 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem7,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem7,200,255,cv2.THRESH_BINARY)
imagem7 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem8,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem8,200,255,cv2.THRESH_BINARY)
imagem8 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem9,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem9,200,255,cv2.THRESH_BINARY)
imagem9 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem10,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem10,200,255,cv2.THRESH_BINARY)
imagem10 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem11,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem11,200,255,cv2.THRESH_BINARY)
imagem11 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem12,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem12,200,255,cv2.THRESH_BINARY)
imagem12 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem13,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem13,200,255,cv2.THRESH_BINARY)
imagem13 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem14,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem14,200,255,cv2.THRESH_BINARY)
imagem14 = thresh1 - thresh2
ret,thresh1=cv2.threshold(imagem15,40,255,cv2.THRESH_BINARY)
ret,thresh2= cv2.threshold(imagem15,200,255,cv2.THRESH_BINARY)
imagem15 = thresh1 - thresh2

X = np.concatenate((imagem1, imagem2, imagem3, imagem4, imagem5,imagem6, imagem7, imagem8, imagem9,imagem10,imagem11,imagem12,imagem13,imagem14,imagem15), axis=0)

y = [0,0,0,0,1,1,0,1,1,1,0,0,1,1,1]

y = np.array(y)

Y = y.reshape(-1)

X = X.reshape(len(y), -1)

classifier_linear = SVC(kernel='linear')

print(40 * '-')
print(' Iniciando o treino do modelo SVC')

classifier_linear.fit(X,Y)


print(40 * '-')



#Tratar as imagens (cinza e resize)
path ="images"

os.chdir(path)
for file in glob.glob("*.jpg"):
    image = cv2.imread(file)
    
    ret,thresh1=cv2.threshold(image,40,255,cv2.THRESH_BINARY)
    ret,thresh2= cv2.threshold(image,200,255,cv2.THRESH_BINARY)
    imagem = thresh1 - thresh2
    
    
    resized = cv2.resize(imagem,(200,200))
    
    prediction = classifier_linear.predict(resized.reshape(1,-1))
    
    if prediction == 1:
        result = 'PossuiParkson'
    elif prediction == 0:
        result = 'NÃOPossuiParkson'
    
    if prediction == 1:
        os.chdir("../Pacientes_Parkson")
        cv2.imwrite(file,resized)
    
    elif prediction == 0:
        os.chdir("../Pacientes_Normal")
        cv2.imwrite(file,resized) 
    
   
    os.chdir("../images")

score = classifier_linear.score(X,Y)
print(score * 100)  
cv2.waitKey(0)
    
    