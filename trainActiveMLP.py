import pickle
import numpy as np
mlpFile = open('Data/mlp.pkl', 'rb')
mlp = pickle.load(mlpFile)

mnistFile = open('Data/mnist.pkl', 'rb')
mnist = pickle.load(mnistFile)

mlpFile.close()
mnistFile.close()

a = np.zeros((len(mnist.train_labels[0]), 10))
for i in range(0, len(mnist.train_labels[0])):
    a[i, int(mnist.train_labels[0, i])] = 1

if mlp.getEpoch() == 0:
    print("Starting MSE: " + str(mlp.costFunction(mnist.getTrain_imgs(), a)))
else:
    print("MSE: " + str(mlp.costFunction(mnist.getTrain_imgs(), a)))
    print("Epoch: " + str(mlp.getEpoch()))
while mlp.costFunction(mnist.getTrain_imgs(), a) > 0:
    mlp.backPropogate(mnist.getTrain_imgs(), a)
    print("MSE: " + str(mlp.costFunction(mnist.getTrain_imgs(), a)))
    print("Epoch: " + str(mlp.getEpoch()))
    output = open('Data/mlp.pkl', 'wb')
    pickle.dump(mlp, output)
    output.close()
