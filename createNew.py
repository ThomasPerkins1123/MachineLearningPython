from MLP import MLP
import pickle

mlp = MLP(784, [300, 200, 100], 10)

output = open('Data/mlp.pkl', 'wb')
pickle.dump(mlp, output)
output.close()


a = 1
