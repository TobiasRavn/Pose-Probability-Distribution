
import math

import numpy as np
import random

class Pose_Accumulator:

    def __init__(self, step_x, x_min, x_max, step_y, y_min, y_max, step_r,training=False, mode = 'all', num = 100, truth=0):
        self._index=-1




        #self.allPoses=[]
        self.allResults=[]
        self.count=0
        self.outputs=[]



        if(mode=='all'):
            x_num=round((x_max-x_min+step_x)/step_x)
            y_num = round((y_max - y_min + step_y) / step_y)
            r_num = round((360) / step_r)
            x_range = np.linspace(x_min,x_max,int(x_num))
            y_range = np.linspace(y_min, x_max, int(y_num))
            r_range = np.linspace(0, 360-step_r, int(r_num))

            self.size=x_num*y_num*r_num
            self.allPoses=np.zeros([self.size,4])

            for x in x_range:
                for y in y_range:
                    for r in r_range:
                        r_rad=math.radians(r)
                        self.allResults.append(0.0)

                        self.allPoses[self.count]=np.array([x,y,math.cos(r_rad),math.sin(r_rad)])
                        self.count += 1
            pass
        elif (mode=='random'):
            self.size=num
            self.allPoses = np.zeros([self.size+1, 4])

            for i in range(self.size):
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                r = random.uniform(0,360)
                r_rad = math.radians(r)
                self.allPoses[i] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])
                self.outputs.append(0)



            x=truth["x"]
            y=truth["y"]
            r=truth["r"]
            x , y, r = float(x), float(y), float(r)
            r_rad = math.radians(r)
            self.outputs.append(1)
            self.allPoses[self.size]=[x, y, math.cos(r_rad), math.sin(r_rad)]
            self.size+=1
            self.count=self.size

    def __iter__(self):
        #return self.allPoses[self._index]
        return self

    def result(self,result):
        self.allResults.append(result)
    def pose(self):
        return self.allPoses[self._index]

    def output(self): #True Output
        return self.outputs[self._index]

    def __next__(self):
        self._index+=1
        if(self._index>=self.count):
            raise StopIteration

        return self
