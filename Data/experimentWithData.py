import numpy as np

# f = open("mnist_train.csv", )

test_data = np.loadtxt("mnist_test.csv", delimiter=",")
f = open("temp.py", "w")

f.write("testData = ")
minX = 0
minY = 0
maxX = 10000
maxY = 785

f.write("[")
for x in range(minX, maxX):
    f.write("[")
    for y in range(minY, maxY):
        f.write(str(int(test_data[x, y])))
        if not y == maxY:
            f.write(", ")
    f.write("]")
    if not x == maxX:
        f.write(", ")
f.write("]")


f.close()
