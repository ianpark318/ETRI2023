import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import math

def Distance2dim(a,b):
    return pow(pow(a[1]-b[1],2)+pow(a[0]-b[0],2), 0.5)

def Cosin2vec(a,b):
    return (a[1]*b[1]+a[0]*b[0])/(pow(pow(a[1],2) + pow(a[0],2) , 0.5)*pow(pow(b[1],2) + pow(b[0],2) , 0.5)) 
def WeightAngle(a,b):
    return math.exp(2*(1.1 - Cosin2vec(a,b)))

def RemoveZero(l):
    nonZeroL = []
    #nonZeroL = []
    for i in range(len(l)):
        if l[i] != 0.0:
            nonZeroL.append(l[i])
    return nonZeroL
#a = [0,-1,0.02,3]
#print RemoveZero(a)

def NormalizeMatrix(_r):
    dimR = _r.shape[0]
    h_max = []
    for i in range(dimR):
        h_max.append(max(_r[i]))
    _max =  max(h_max)
    h_min = []
    for i in range(dimR):
        #print _r[i]
#         h_min.append(min(RemoveZero(_r[i])))
        h_min.append(min(_r[i]))
    
    _min =  min(h_min)
    _max_min = _max - _min
    _normalizedRP = np.zeros((dimR,dimR))
    for i in range(dimR):
        for j in range(dimR):
            _normalizedRP[i][j] = (_r[i][j]-_min)/_max_min
    return _normalizedRP

def varRP(length, data, dim):#dim:=x,y,z
    x = []
    if dim == 'x':
        for j in range(length):
            x.append(data.iloc()[j][1])
    elif dim == 'y':
        for j in range(length):
            x.append(data.iloc()[j][2])
    elif dim == 'z':
        for j in range(length):
            x.append(data.iloc()[j][3])
    
    s = []
    for i in range(len(x)-1):
        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
        
    #print s
    dimR = len(x)-1
    #R = np.zeros((dimR,dimR))
    R = np.eye(dimR)
    for i in range(dimR):
        for j in range(dimR):
            if Cosin2vec(list(map(lambda x:x[0]-x[1], zip(s[i], s[j]))), [1,1])>= pow(2, 0.5)/2:
                sign =1.0
            else:
                sign =-1.0
            R[i][j] = sign*Distance2dim(s[i],s[j])
    return R

def RP(data, dim):#dim:=x,y,z
    x = []
    if dim == 'x':
        for j in range(length):
            x.append(data[j][1])
    elif dim == 'y':
        for j in range(length):
            x.append(data[j][2])
    elif dim == 'z':
        for j in range(length):
            x.append(data[j][3])
    
    s = []
    for i in range(len(x)-1):
        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
        
    #print s
    dimR = len(x)-1
    R = np.zeros((dimR,dimR))

    for i in range(dimR):
        for j in range(dimR):
            R[i][j] = Distance2dim(s[i],s[j])
    return R

def RGBfromRPMatrix_of_XYZ(X,Y,Z):
    if X.shape != Y.shape or X.shape != Z.shape or Y.shape != Z.shape:
        print('XYZ should be in same shape!')
        return 0
    
    dimImage = X.shape[0]
    newImage = np.zeros((dimImage,dimImage,3))
    for i in range(dimImage):
        for j in range(dimImage):
            _pixel = []
            _pixel.append(X[i][j])
            _pixel.append(Y[i][j])
            _pixel.append(Z[i][j])
            newImage[i][j] = _pixel
    return newImage

def SaveRP(x_array,y_array,z_array):
    _r = RP(x_array)
    _g = RP(y_array)
    _b = RP(z_array)
    plt.close('all')
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
    plt.imshow(newImage)
    plt.savefig('D:\Datasets\ADL_Dataset\\'+action+'\\'+'RP\\''{}{}.png' .format(action, subject[15:]),bbox_inches='tight',pad_inches = 0)
    plt.close('all')
    
def SaveRP_XYZ(x,action, normalized):
    _r = RP(x,'x')
    _g = RP(x,'y')
    _b = RP(x,'z')
    plt.close('all')
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if normalized:
        newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(NEWPATH+action+'\\RP\\'+'{}{}.png'  .format(action, i),bbox_inches='tight',pad_inches = 0)
        plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(NEWPATH+action+'\\RP\\'+'{}{}.png'  .format(action, i),bbox_inches='tight',pad_inches = 0)
        plt.close('all')

def SavevarRP_XYZ(length, x, action, normalized):
    _r = varRP(length, x, 'x')
    _g = varRP(length, x, 'y')
    _b = varRP(length, x, 'z')
    plt.close('all')
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if normalized:
        newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(action+'{}.png'  .format(i[:-4]),bbox_inches='tight',pad_inches = 0)
        plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(action+'{}.png'  .format(i[:-4]),bbox_inches='tight',pad_inches = 0)
        plt.close('all')
        
path = '../ETRI/data/dataset_2020/user03'
user_path = os.listdir(path)

for date in user_path:
    user_n_data_path = os.listdir(path + '/' + date + '/e4Acc')
    for i in user_n_data_path:
        if len(os.listdir(path + '/' + date + '/e4Acc')) == 0:
            continue
        else:
            data = pd.read_csv(path + '/' + date + '/e4Acc/' + i)
            if os.path.exists('train/' + 'user03' + '/' + date):
                pass
            else:
                os.mkdir('train/' + 'user03' + '/' + date)
                os.mkdir('train/' + 'user03' + '/' + date + '/RP')
                
            SavevarRP_XYZ(len(data), data, 'train/' + 'user03' + '/' + date + '/RP/', normalized=1)