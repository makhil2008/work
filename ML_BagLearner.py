
import numpy as np

import DTLearner as dt
import RTLearner as rt
import LinRegLearner as lrl

class BagLearner(object):
   """
   This is a Linear Regression Learner. It is implemented correctly.

   :param verbose: If “verbose” is True, your code can print out information for debugging.
       If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
   :type verbose: bool
   """
   def __init__(self,learner,boost,verbose,bags,kwargs):
       """
       Constructor method
       """
       self.learner=learner
       self.bags=bags
       self.leaf_size = kwargs.get('leaf_size',1)
       self.learners = []
       for i in range(0,self.bags):
           self.learners.append(self.learner(**kwargs))
       pass  # move along, these aren't the drones you're looking for

   def author(self):
       """
       :return: The GT username of the student
       :rtype: str
       """
       return "AMANEPALLI7"  # replace tb34 with your Georgia Tech username

   def add_evidence(self, data_x, data_y):
       """
       Add training data to learner

       :param data_x: A set of feature values used to train the learner
       :type data_x: numpy.ndarray
       :param data_y: The value we are attempting to predict given the X data
       :type data_y: numpy.ndarray
       """
       for i in range(0,self.bags):
           random_rows=np.random.randint(data_x.shape[0],size=data_x.shape[0])
           this_x=data_x[random_rows,:]
           this_y=data_y[random_rows]
           self.learners[i].add_evidence(this_x,this_y)



   def query(self, points):
       """
       Estimate a set of test points given the model we built.

       :param points: A numpy array with each row corresponding to a specific query.
       :type points: numpy.ndarray
       :return: The predicted result of the input data according to the trained model
       :rtype: numpy.ndarray
       """
       ypreds=np.empty([points.shape[0],self.bags])
       for i in range(0,self.bags):
           ypreds[:,i]=np.array(self.learners[i].query(points))
       return np.mean(ypreds,axis=1)




if __name__ == "__main__":
   print("the secret clue is 'zzyzx'")
