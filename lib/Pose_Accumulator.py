
import math



class Pose_Accumulator:

    def __init__(self, step_x, x_min, x_max, step_y, y_min, y_max, step_r):
        self._index=0
        self.allPoses=[]
        self.allResults=[]
        self.count=0
        for x in range(x_min,x_max,step_x):
            for y in range(y_min, y_max, step_y):
                for r in range(0, 360, step_r):
                    r_rad=math.radians(r)
                    self.allResults.append(0.0)
                    self.count+=1
                    self.allPoses.append([x,y,math.cos(r_rad),math.sin(r_rad)])

        pass

    def __iter__(self):
        return self.allPoses[self._index]
        return self

    def result(self,result):
        self.allResults[self._index]=result

    def __next__(self):
        self._index+=1
        if(self._index>=self.count):
            raise StopIteration


