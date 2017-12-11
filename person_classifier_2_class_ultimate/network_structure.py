# python library
import numpy as np

# chainer library
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import reporter


# networks
class CNN_thibault_ultimate(chainer.Chain):

    def __init__(self, train= True):
        super(CNN_thibault_ultimate, self).__init__(
            conv1 = L.Convolution2D(3, 6, 3, stride=1),
            conv2 = L.Convolution2D(6, 6, 3, pad=1),
            conv3 = L.Convolution2D(6, 6, 3, pad=1),
            l1 = L.Linear(486, 432),
            l2 = L.Linear(432, 100),
            lo = L.Linear(100, 2),
            bn1 = L.BatchNormalization(6),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        y = self.lo(h)
        return y

    def __call__(self, x, t):
        self.clear()
        y = self.forward(x)
        self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss

#model5
class CNN_thibault2(chainer.Chain):

    def __init__(self, train= True):
        super(CNN_thibault2, self).__init__(
            conv1 = L.Convolution2D(3, 6, 3, stride=1),
            conv2 = L.Convolution2D(6, 6, 3, pad=1),
            l1 = L.Linear(864, 432),
            l2 = L.Linear(432, 100),
            lo = L.Linear(100, 2),
            bn1 = L.BatchNormalization(6),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        y = self.lo(h)
        return y

    def __call__(self, x, t):
        self.clear()
        y = self.forward(x)
        self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
