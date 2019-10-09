#coding: UTF-8
# Numpy
import numpy as np
# Chainer
import chainer
import math
import random
from chainer import Chain, Variable
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
'''
'''
class NeuralNet(Chain):
    def __init__(self):
        # self.input_num=9
        self.output_data=[]
        super(NeuralNet, self).__init__(
            l1 = L.Linear(None, 7),
            # bn1 = L.BatchNormalization(7),
            l2 = L.Linear(None,2)
        )
    #推論
    def forward(self,input_data):
        h=F.relu(self.l1(input_data))
        self.output_data=F.relu(self.l2(h))
        # self.output_data=F.softmax(h)
        return self.output_data

    # Loss
    def loss(self,x,t):
        self.output_data=self.forward(x)
        loss=F.softmax_cross_entropy(self.output_data, t)
        # loss=Variable(np.array(loss))
        return loss
    # Accuracy
    def acc(self,t):
        return F.accuracy(self.output_data,t)
