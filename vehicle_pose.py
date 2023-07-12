import numpy as np


class pose:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        
    def update(self,v, gamma, delta_t):
        '''
        theta: heading angle
        v: longitudinal velocity
        gamma: yaw rate
        delta_t: time interval
        '''
        self.x = self.x + v * delta_t * np.cos(np.radians(self.theta))
        self.y = self.y + v * delta_t * np.sin(np.radians(self.theta))
        self.theta = self.theta + gamma * delta_t
        
    def get_pose(self):
        return np.array([self.x, self.y, self.theta])
    