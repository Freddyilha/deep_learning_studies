
import random
import cPickle
import numpy as np
from PIL import Image
import scipy

nb_samples = 50000
X = np.zeros((nb_samples, 3, 32, 32), dtype="uint8")
y = np.zeros((nb_samples,))
for i in range(1, 6):
    fpath = 'data_batch_' + str(i)
    f = open(fpath, 'rb')
    d = cPickle.load(f)
    f.close()
    data = d["data"]
    labels = d["labels"]
    
    data = data.reshape(data.shape[0], 3, 32, 32)
    X[(i-1)*10000:i*10000, :, :, :] = data
    y[(i-1)*10000:i*10000] = labels
    

for i in range(0, 50000):
    pix = [0] * 1024
    #img = np.zeros([32,32])
    cont = 0
    for j in range(0, 32):
        for k in range(0, 32):
            pix[cont] = int(X[i][0][j][k]*0.299 + X[i][1][j][k]*0.587 + X[i][2][j][k]*0.114)
            #img[j][k] = int(X[i][0][j][k]*0.299 + X[i][1][j][k]*0.587 + X[i][2][j][k]*0.114)
            cont += 1

    #im = Image.fromarray(img)
    #im.convert('L').save('outfile' + str(i) + '.png')
    print (pix, int(y[i]))
