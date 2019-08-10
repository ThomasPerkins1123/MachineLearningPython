import numpy as np


def printList(list):
    for x in list:
        print(x)


print("hello world")
arrayin = np.zeros((2,1))
array = np.zeros((2, 4))
array[1][1] = 1

print("in")
printList(arrayin)
print("mid")
printList(array)
