import pickle
import numpy as np
import mnist

mlpFile = open('Data/mlp.pkl', 'rb')
mlp = pickle.load(mlpFile)

mlpFile.close()

images = mnist.test_images()
images = images/255

#for image in images:
#    for x in image:
#        for y in x:
#            images[image][x][y] = images[image][x][y]/256
            

for image in images:
    for x in image:
        for y in x:
            print(y)
            if y > 0:
                if y > 1:
                    print("1", end="")
                else:
                    print("#", end="")
            else:
                print(" ", end="")
        print()

