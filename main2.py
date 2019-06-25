# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:17:44 2019

@author: kayo_
"""
import cv2
import numpy as np
from sklearn.svm import SVC
'''from sklearn.svm import SVR
import pylab as pl'''

# Read all images
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
testimagem2_g = cv2.imread('images/testimagem2.jpg')
testimagem_g = cv2.imread('images/testimagem.jpg')
result1_g = cv2.imread('images/result1.png')
result0_g = cv2.imread('images/result0.png')

'''
ret,thresh3=cv2.threshold(testimagem_g,40,255,cv2.THRESH_BINARY)
ret,thresh4= cv2.threshold(testimagem_g,200,255,cv2.THRESH_BINARY)
teste1 = thresh3 - thresh4'''


# Resize images to 10px x 10px

'''scale_percent = 60 # percent of original size
width = int(imagem1_g.shape[1] * scale_percent / 100)
height = int(imagem1_g.shape[0] * scale_percent / 100)
dim = (width, height)'''

imagem1 = cv2.resize(imagem1_g, (10,10),interpolation = cv2.INTER_AREA)
imagem2 = cv2.resize(imagem2_g, (10,10),interpolation = cv2.INTER_AREA)
imagem3 = cv2.resize(imagem3_g, (10,10),interpolation = cv2.INTER_AREA)
imagem4 = cv2.resize(imagem4_g, (10,10),interpolation = cv2.INTER_AREA)
imagem5 = cv2.resize(imagem5_g, (10,10),interpolation = cv2.INTER_AREA)
imagem6 = cv2.resize(imagem6_g, (10,10),interpolation = cv2.INTER_AREA)
imagem7 = cv2.resize(imagem7_g, (10,10),interpolation = cv2.INTER_AREA)
imagem8 = cv2.resize(imagem8_g, (10,10),interpolation = cv2.INTER_AREA)
imagem9 = cv2.resize(imagem9_g, (10,10),interpolation = cv2.INTER_AREA)
imagem10 = cv2.resize(imagem10_g, (10,10),interpolation = cv2.INTER_AREA)
testimagem2 = cv2.resize(testimagem2_g, (10,10),interpolation = cv2.INTER_AREA)
testimagem = cv2.resize(testimagem_g, (10,10),interpolation = cv2.INTER_AREA)

#TECNICA DE LIMIARIZAÇÃO + OPERAÇÃO DE SUBITRAÇÃO
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

# Imagens que serão testadas
ret,thresh1=cv2.threshold(testimagem2,40,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(testimagem2,200,255,cv2.THRESH_BINARY)
testimagem2 = thresh1 - thresh2
ret,thresh1=cv2.threshold(testimagem,40,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(testimagem,200,255,cv2.THRESH_BINARY)
testimagem = thresh1 - thresh2




# Concat all arrays to one
X = np.concatenate((imagem1, imagem2, imagem3, imagem4, imagem5,imagem6, imagem7, imagem8, imagem9,imagem10), axis=0)



# Create index to arrays
y = [0,0,0,0,1,1,0,1,1,1]

# Set y as a array
y = np.array(y)

# Reshape y
Y = y.reshape(-1)

# Reshape X with length of y
X = X.reshape(len(y), -1)

# Create the classifier 
classifier_linear = SVC(kernel='linear')

print(40 * '-')
print(' Iniciando o treino do modelo SVC')

# Train the classifier with images and indexes
classifier_linear.fit(X,Y)


print(40 * '-')



# Predict the category of image 
prediction = classifier_linear.predict(testimagem.reshape(1,-1))


# Score of predict 
score = classifier_linear.score(X,Y)

# Show prediction
print('Resultado: {}'.format(prediction))

# Show prediction score
'''print('Score of precision: {:.1f}%'.format(score * 100))'''    
if prediction == 1:
	result = 'Possui Parkson'
elif prediction == 0:
	result = 'NÃO Possui Parkson'
# Show image based on prediction
'''cv2.imshow("Result", result)'''
print(result)


# Show the image tested
'''cv2.imshow("Test2", thresh1)
cv2.imshow("Test3", thresh2)'''
if prediction == 1:
    cv2.imwrite("C:/Users/kayo_/Desktop/IFAL/parkson_ifal/Pacientes_Parkson/parkson1.jpg",testimagem_g)
elif prediction == 0:
    cv2.imwrite("C:/Users/kayo_/Desktop/IFAL/parkson_ifal/Pacientes_Normal/normal1.jpg",testimagem_g)  
testimagem_r = cv2.resize(testimagem_g, (250,250),interpolation = cv2.INTER_AREA)
cv2.imshow("Imagem Utilizada para realizar o teste", testimagem_r)
'''   
cv2.imshow("Test1", thresh3)
cv2.imshow("Test2", thresh4)
cv2.imshow("Test3", teste1)'''
# Wait for key
cv2.waitKey(0)














print('---------------------------------------')

'''
# Create the classifier 
classifier_linear_regression = SVR(kernel='linear')

print('Start SVR Train')

# Train the classifier with images and indexes
classifier_linear_regression.fit(X,Y)

print('Finished train')
print(40 * '-')

# Predict the category of image 
prediction = classifier_linear_regression.predict(park5.reshape(1,-1))

# Score of predict 
score = classifier_linear_regression.score(X,Y)

# Show prediction
print('Result: {}'.format(prediction))

# Show prediction score
print('Score of precision: {:.1f}%'.format(score * 100))

# Set result as image of prediction
if prediction == 1:
	result = normal1_g
elif prediction == 2:
	result = normal2_g
elif prediction == 3:
	result = normal3_g
elif prediction == 4:
	result = normal4_g
elif prediction == 5:
	result = normal5_g
elif prediction == 6:
	result = park1_g
elif prediction == 7:
	result = park2_g
elif prediction == 8:
	result = park3_g
elif prediction == 9:
	result = park4_g
elif prediction == 10:
	result = park5_g

# Show image based on prediction
cv2.imshow("Result", result)
# Show the image tested
cv2.imshow("Test", park5_g)
# Wait for key
cv2.waitKey(0)'''