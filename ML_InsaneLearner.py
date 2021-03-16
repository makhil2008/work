import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
   def __init__(self,verbose):
       self.learners = [bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False) for i in range(0,20)]
       pass
   def author(self):
       return "AMANEPALLI7"
   def add_evidence(self, data_x, data_y):
       [self.learners[i].add_evidence(data_x,data_y) for i in range(0,20)]
   def query(self, points):
       ypreds=[np.array(self.learners[i].query(points)) for i in range(0,20)]
       return np.mean(ypreds,axis=1)
if __name__ == "__main__":
   print("the secret clue is 'zzyzx'")
