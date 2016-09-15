#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#GABBAR STUDIOWORKS
#
# Copyright © 2016 gabbar1947 <gabbar1947@Rathores-MacBook-Pro.local>
#
# Distributed under terms of the MIT license.

import numpy as np

lr = 0.001;

TF = 'trainingdata.tra'
with open(TF) as k:
    tra_ori = k.readlines()



tra_list = [map(int, item.strip().split(',')) for item in tra_ori]
tra_data = np.transpose(np.asmatrix([item[:-1] for item in tra_list]))
tra_gt = [item[-1] for item in tra_list]
mean_sample = np.divide(tra_data.sum(axis = 1), tra_data.shape[1])
tra_data = np.subtract(tra_data, mean_sample)

def sigmoid(x):
    x = x*-1;
    x = np.exp(x);
    x = x+1.0;
    return np.divide(1.0,x);

class BCP_lyr:

    #layer initialization
    def __init__(self, num , n, prev_n):
        self.num = num;
        self.nl = n;
        self.prevO = np.random.rand(prev_n,1);
        self.w = np.random.rand(n, prev_n);
        self.bias = np.random.rand(n,1);
        #print self.w.shape, n, prev_n
        self.net = (self.w).dot(self.prevO) + self.bias;
        self.func = sigmoid(self.net);
        #print self.prevO.shape[1];
        self.deriv = np.zeros(((self.prevO).shape[0],(self.prevO).shape[1]));

    def forward(self, p_out):
        self.prevO = p_out;
        self.net = (self.w).dot(self.prevO) + (self.bias);
        
        if self.num == 1:
            self.w = np.identity((self.w).shape[0]);
            self.bias = np.zeros(((self.w).shape[0], (self.w).shape[1]));
            self.net = (self.w).dot(self.prevO) + (self.bias);
            self.func = self.prevO;
        else:
            self.func = sigmoid(self.net);
    

    def back(self,next_deriv,next_weight=None):
        if self.num == 3:
            dc_o = self.func - next_deriv;
            dc_net = np.multiply(dc_o, np.multiply(sigmoid(self.net),(1-sigmoid(self.net))));
            self.deriv = dc_net;
        else:
            dc_o = np.transpose(next_weight)*next_deriv;
            dc_net = np.multiply(dc_o, np.multiply(sigmoid(self.net),(1-sigmoid(self.net))));
            self.deriv = dc_net;
            # update weights by adding -1*lr*derivative
            # update bias "   "  "       "    "

        DW = dc_net*np.transpose(self.prevO)*(-1)*(lr);
        self.w = self.w + DW;
        self.bias = self.bias + (-lr)*dc_net;

def main():
    L1 = BCP_lyr(1 , 64 , 64);
    L2 = BCP_lyr(2 , 20, L1.nl);
    L3 = BCP_lyr(3 , 10, L2.nl);
    for epoch in range(0,200):
        #print "hello"
        count=0;
        for i in range(0,tra_data.shape[1]):

            L1.forward( tra_data[:,i]);
            L2.forward( L1.func);
            L3.forward( L2.func);
            
            #target vector
            
            target=np.zeros((10,1));
            target[tra_gt[i]] = 1;

            #back prop
        
            L3.back(target);
            L2.back(L3.deriv,L3.w);

            index = np.argmax(L3.func);
            if index == tra_gt[i]:
                count = count+1;

        if epoch%5 == 0:
            error = float(tra_data.shape[1]- count)/tra_data.shape[1];
            print "hello my name is",error

main();
