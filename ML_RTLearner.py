
import numpy as np


class RTLearner(object):
   """
   This is a Linear Regression Learner. It is implemented correctly.

   :param verbose: If “verbose” is True, your code can print out information for debugging.
       If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
   :type verbose: bool
   """
   def __init__(self, leaf_size,verbose=False):
       """
       Constructor method
       """
       self.leaf_size=leaf_size
       self.model=[]
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

       # slap on 1s column so linear regression finds a constant term
       if(data_y.shape[0]<=self.leaf_size or all(x==data_y[0] for x in data_y)):
           self.model.insert(0,[-1, -1,-1,np.mean(data_y)])
           return 1
       else:
           new_data_x = data_x
           np.seterr(invalid='ignore')
           split_var=np.random.randint(new_data_x.shape[1], size=1)[0]
           split_val=np.median(new_data_x[:,split_var])
           if (split_val==np.min(new_data_x[:,split_var]) or split_val==np.max(new_data_x[:,split_var])):
               self.model.insert(0,[-1, -1,-1,np.mean(data_y)])
               return 1
           left_X=new_data_x[new_data_x[:,split_var]<=split_val]
           left_y=data_y[new_data_x[:,split_var]<=split_val]
           right_X=new_data_x[new_data_x[:,split_var]>split_val]
           right_y=data_y[new_data_x[:,split_var]>split_val]
           right_length=self.add_evidence(right_X,right_y)
           left_length=self.add_evidence(left_X,left_y)
           self.model.insert(0,[split_var, split_val,1,left_length+1])
           sub_tree_length=left_length+right_length

           return sub_tree_length+1

   def gety(self,onepoint):
       thismodel=np.array(self.model)
       next=True
       row_num=0
       while next==True:
           row_num=int(row_num)
           split_var=int(thismodel[row_num,0])
           split_val=thismodel[row_num,1]
           if split_val==-1 and split_val==-1:
               next=False
               ypred=thismodel[row_num,3]
           elif onepoint[split_var]<=split_val:
               row_num=row_num+1
           elif onepoint[split_var]>split_val:
               row_num=row_num+thismodel[row_num,3]
       return ypred

   def query(self, points):
       """
       Estimate a set of test points given the model we built.

       :param points: A numpy array with each row corresponding to a specific query.
       :type points: numpy.ndarray
       :return: The predicted result of the input data according to the trained model
       :rtype: numpy.ndarray
       """

       return [self.gety(points[int(i),:]) for i in range(0,points.shape[0])]




if __name__ == "__main__":
   print("the secret clue is 'zzyzx'")
