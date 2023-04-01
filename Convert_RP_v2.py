import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def Distance2dim(a,b):
    return pow(pow(a[1]-b[1],2)+pow(a[0]-b[0],2), 0.5)

def Cosin2vec(a,b):
    x = (a[1]*b[1]+a[0]*b[0])/(pow(pow(a[1],2) + pow(a[0],2) , 0.5)*pow(pow(b[1],2) + pow(b[0],2) , 0.5)) 
    return x
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

def NormalizeMatrix(_r):
    dimR = _r.shape[0]
    h_max = []
    for i in range(dimR):
        h_max.append(max(_r[i]))
    _max =  max(h_max)
    h_min = []
    for i in range(dimR):
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

def SavevarRP_XYZ(length, x, path, normalized, num):
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
        plt.imshow(newImage)
        plt.savefig(path+ str(num)[:-2] + '.png' ,bbox_inches='tight',pad_inches = 0)
        plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(path+ str(num)[:-2] + '.png' ,bbox_inches='tight',pad_inches = 0)
        plt.close('all')
        

## Read Directory and Convert 3 axis e4Acc to 3 channel image
path = 'user01-06/'
user_path = os.listdir('./user01-06/')

action_list = ['personal_care', 'sleep', 'study', 'hobby', 'shop', 'community_interaction', 'entertainment']

for user_n in user_path:
    if user_n != 'user06':
        continue
    user_n_path = os.listdir(path + user_n + '/')
    
    for date in user_n_path:
        if len(os.listdir(path + user_n + '/' + date + '/e4Acc')) == 0:
                continue
        action_label = pd.read_csv(path + user_n + '/' + date + '/' + date + '_label.csv')
        mk_action_data = action_label.drop(labels=range(0, len(action_label)), axis=0)
        
        for class_list in action_list:
            tmp_data = action_label[action_label['action'].str.contains(class_list)]
            mk_action_data = pd.concat([mk_action_data, tmp_data])
            
        for i, action in zip(mk_action_data['ts'], mk_action_data['action']):
            num = str(i)
            print('Make ', str(i)[:-2], ' file', 'action is ', action)
            
            if os.path.exists(path + user_n + '/' + date + '/e4Acc/' + str(i)[:-2] + '.csv'):    
                data = pd.read_csv(path + user_n + '/' + date + '/e4Acc/' + str(i)[:-2] + '.csv')
                
                if os.path.exists('train/data/' + user_n + '/' + date + '/RP'):
                    # if len(os.listdir('train/data' + user_n + '/' + date + '/RP/')) == len(os.listdir(path + user_n + '/' + date + '/e4Acc')):
                    #     continue
                    pass
                else:
                    os.mkdir('train/data/' + user_n + '/' + date)
                    os.mkdir('train/data/' + user_n + '/' + date + '/RP')
                    
                SavevarRP_XYZ(len(data), data, 'train/data/' + user_n + '/' + date + '/RP/', normalized=True, num=num)
                
            else:
                pass