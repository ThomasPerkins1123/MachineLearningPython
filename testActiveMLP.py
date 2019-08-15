import pickle
import numpy as np
mlpFile = open('Data/mlp.pkl', 'rb')
mlp = pickle.load(mlpFile)

mnistFile = open('Data/mnist.pkl', 'rb')
mnist = pickle.load(mnistFile)

mlpFile.close()
mnistFile.close()

a = np.zeros((len(mnist.test_labels[0]), 10))
for i in range(0, len(mnist.test_labels[0])):
    a[i, int(mnist.test_labels[0, i])] = 1

b = mlp.forwardPropogate
print(mlp.costFunction(mnist.getTest_imgs(), a))
mlp.backPropogate(mnist.getTest_imgs(), a)
print(mlp.costFunction(mnist.getTest_imgs(), a))
# mlp.forwardPropogate(mnist.getTrain_imgs())


# mlp.costFunction(mnist.getTest_imgs(), a)

