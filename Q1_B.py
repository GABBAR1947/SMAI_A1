#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# GABBAR STUDIOWORKS
#
# Copyright © 2016 gabbar1947 <gabbar1947@Rathores-MacBook-Pro.local>
#
# Distributed under terms of the MIT license.

import numpy as np
import matplotlib.pyplot as pt

A = np.array([[2,7],[8,1],[7,5],[6,3],[7,8],[5,9],[4,5]]);

B = np.array([[4,2],[-1,-1],[1,3],[3,-2],[5,3.25],[2,4],[7,1]]);

y = np.array([
    [1,2,7],[1,8,1],[1,7,5],[1,6,3],[1,7,8],[1,5,9],[1,4,5],
    [-1,-4,-2],[-1,1,1],[-1,-1,-3],[-1,-3,2],[-1,-5,-3.25],[-1,-2,-4],[-1,-7,-1]
    ]);

a = np.ones(3);
a[0]=-0.001;
count=0;

i=0;b=4;
while i<14:
    x = y[i].dot(a);
    print x,
    if x<=4:
        count=count+1;
        a = a + y[i];
    
    print a
    if i == 13 and count == 0:
        break;

    if i==13:
        i=-1;
        count=0;
    i=i+1;

A=A.transpose();
B=B.transpose();

pt.scatter(B[0],B[1],color='blue',s=2);
pt.scatter(A[0],A[1],color='red',s=2);
pt.legend(['Class W2','Class W1'],loc='upper left');


x1=11;
x2=-11;
y1 = ((b-a[0]-a[1]*x1)/a[2]);
y2 = ((b-a[0]-a[1]*x2)/a[2]);

print y1,y2

pt.plot([x1,x2],[y1,y2],linewidth=0.25);
pt.show();
