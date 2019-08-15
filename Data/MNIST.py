class MNIST:
    train_imgs = 0
    test_imgs = 0
    train_labels = 0
    test_labels = 0

    def __init__(self, a, b, c, d):
        self.train_imgs = a
        self.test_imgs = b
        self.train_labels = c
        self.test_labels = d

    def getTrain_imgs(self):
        return self.train_imgs

    def getTest_imgs(self):
        return self.test_imgs

    def getTrain_labels(self):
        return self.train_labels

    def getTest_labels(self):
        return self.test_labels
