from MLP import MLP
import pickle

# f = open("Data/mlp.py", "w")

mlp = MLP(784, [300, 200, 100], 10)

# f.write("E = " + str(mlp.E) + "\n")
# f.write("hiddenLayer_W = " + str( + "\n")
# f.close()

output = open('Data/mlp.pkl', 'wb')
pickle.dump(mlp, output)
output.close()


a = 1
