import numpy as np
np.random.seed(1)

def sigmoid(z):
  """
  sigmoid function that maps inputs into the interval [0,1]
  Your implementation must be able to handle the case when z is a vector (see unit test)
  Inputs:
  - z: a scalar (real number) or a vector
  Outputs:
  - trans_z: the same shape as z, with sigmoid applied to each element of z
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  trans_z = list(map(lambda x: 1/(1 + np.exp(-x)) , z)) 
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return trans_z

def logistic_regression(X, w):
  """
  logistic regression model that outputs probabilities of positive examples
  Inputs:
  - X: an array of shape (num_sample, num_features)
  - w: an array of shape (num_features,)
  Outputs:
  - logits: a vector of shape (num_samples,)
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  logits = list(map(lambda x: 1/(1 + np.exp(-np.sum(x * w))) , X))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return logits


def logistic_loss(X, w, y):
  """
  a function that compute the loss value for the given dataset (X, y) and parameter w;
  It also returns the gradient of loss function w.r.t w
  Here (X, y) can be a set of examples, not just one example.
  Inputs:
  - X: an array of shape (num_sample, num_features)
  - w: an array of shape (num_features,)
  - y: an array of shape (num_sample,), it is the ground truth label of data X
  Output:
  - loss: a scalar which is the value of loss function for the given data and parameters
  - grad: an array of shape (num_featues,), the gradient of loss 
  """
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  num_samples=len(X)
  z= np.dot(X,w)
  sigmo=sigmoid(z)
  class1_loss= -1*y*np.log(sigmo)
  class2_loss=(1-y)*np.log(1-np.array(sigmo))
  loss=class1_loss-class2_loss
  loss=loss.sum()/num_samples
  
  grad=np.dot(X.T,sigmo-y)/num_samples
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return loss, grad

def softmax(x):
  """
  Convert logits for each possible outcomes to probability values.
  In this function, we assume the input x is a 2D matrix of shape (num_sample, num_classes).
  So we need to normalize each row by applying the softmax function.
  Inputs:
  - x: an array of shape (num_sample, num_classse) which contains the logits for each input
  Outputs:
  - probability: an array of shape (num_sample, num_classes) which contains the
                probability values of each class for each input
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  num= np.exp(x)
  deno= np.sum(num,axis=1)
  probability=(num.T/deno).T
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return probability

def MLR(X, W):
  """
  performs logistic regression on given inputs X
  Inputs:
  - X: an array of shape (num_sample, num_feature)
  - W: an array of shape (num_feature, num_class)
  Outputs:
  - probability: an array of shape (num_sample, num_classes)
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  z= np.dot(X,W)
  probability= softmax(z)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return probability

def cross_entropy_loss(X, W, y):
  """
  Inputs:
  - X: an array of shape (num_sample, num_feature)
  - W: an array of shape (num_feature, num_class)
  - y: an array of shape (num_sample,)
  Ouputs:
  - loss: a scalar which is the value of loss function for the given data and parameters
  - grad: an array of shape (num_featues, num_class), the gradient of the loss function 
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  num_samples=len(y)
  z= np.dot(X,W)
  sigmo=softmax(z) 
  loss=np.sum(-np.log(sigmo[np.arange(num_samples),y]))/num_samples
  sigmo[np.arange(num_samples),y] -= 1
  sigmo /= num_samples
  grad = X.T.dot(sigmo)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return loss, grad

def gini_score(groups, classes):
  '''
  Inputs: 
  groups: 2 lists of examples. Each example is a list, where the last element is the label.
  classes: a list of different class labels (it's simply [0.0, 1.0] in this problem)
  Outputs:
  gini: gini score, a real number
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  # count all samples at split point
  from collections import Counter 
#   count all samples at split point 

  g1_len=len(groups[0])
  g2_len= len(groups[1])
  total_samples=g1_len+g2_len

  if(g1_len==0): 
        g1_score=0.0
  else:
      g1_labels= [i[-1] for i in groups[0]]
      g1_counter= list(Counter(g1_labels).items())
      g1_score=1- np.sum(list(map(lambda x:(x[1]/g1_len)**2,g1_counter)))
        
        
  if(g2_len==0):
    g2_score=0.0
  else:
      g2_labels=[i[-1] for i in groups[1]]
      g2_counter= list(Counter(g2_labels).items())
      g2_score=1-np.sum(list(map(lambda x:(x[1]/g2_len)**2,g2_counter)))
  
  gini=((g1_len/total_samples)* g1_score) + ((g2_len/total_samples)*g2_score)
  
  # sum weighted Gini index for each group

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return gini

def create_split(index, threshold, datalist):
  '''
  Inputs:
  index: The index of the feature used to split data. It starts from 0.
  threshold: The threshold for the given feature based on which to split the data.
        If an example's feature value is < threshold, then it goes to the left group.
        Otherwise (>= threshold), it goes to the right group.
  datalist: A list of samples. 
  Outputs:
  left: List of samples
  right: List of samples
  '''
  
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  left = []
  right = []
  for data in datalist:
     if(data[index] < threshold):
            left.append(data)
     else:
            right.append(data)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return left, right

def get_best_split(datalist):
  '''
  Inputs:
  datalist: A list of samples. Each sample is a list, the last element is the label.
  Outputs:
  node: A dictionary contains 3 key value pairs, such as: node = {'index': integer, 'value': float, 'groups': a tuple contains two lists of examples}
  Pseudo-code:
  for index in range(#feature): # index is the feature index
    for example in datalist:
      use create_split with (index, example[index]) to divide datalist into two groups
      compute the Gini index for this division
  construct a node with the (index, example[index], groups) that corresponds to the lowest Gini index
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  classes=[set(i[-1] for i in datalist)]
  min_sofar=9999999.0
  node={'index':-1,"value":-1.0,"groups":()}
  for index in range(len(datalist[0])-1):
    for example in datalist:
        result=create_split(index,example[index],datalist)
#         print(result)
        gini_result=gini_score(result, classes)
#         print(example[index],gini_result, index)
        if(min_sofar>gini_result):
            min_sofar=gini_result
            node['index']=index
            node['value']=example[index]
            node['groups']=result
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return node

def to_terminal(group):
  '''
  Input:
    group: A list of examples. Each example is a list, whose last element is the label.
  Output:
    label: the label indicating the most common class value in the group
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  from collections import Counter 
  result=Counter([i[-1] for i in group])
#   print(result)
  label=result.most_common()[0][0]
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return label


def recursive_split(node, max_depth, min_size, depth):
  '''
  Inputs:
  node:  A dictionary contains 3 key value pairs, node = 
         {'index': integer, 'value': float, 'groups': a tuple contains two lists fo samples}
  max_depth: maximum depth of the tree, an integer
  min_size: minimum size of a group, an integer
  depth: tree depth for current node
  Output:
  no need to output anything, the input node should carry its own subtree once this function ternimate
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  left_g, right_g=node['groups']
  del(node['groups'])

    # if either left_g or right_g is empty:
  if(not left_g or not right_g):
    node['left']=to_terminal(left_g+right_g)
    node['right']= node['left']
    return

 
# # check for max depth
# if depth of N >= max_depth - 1:   # use >= instead of == in case max_depth = 1
  if(depth>=max_depth-1):
        node['left']=to_terminal(left_g)
        node['right']=to_terminal(right_g)
        return

 
# # process left child
# if the number of examples in left_g <= min_size:
  if(len(left_g)<=min_size):
        node['left']= to_terminal(left_g)
  else:
    node['left']= get_best_split(left_g)
    recursive_split(node['left'],max_depth, min_size,depth+1)
    
  if(len(right_g)<=min_size):
    node['right']= to_terminal(right_g)
  else:
    node['right']= get_best_split(right_g)
    recursive_split(node['right'],max_depth, min_size,depth+1)
  

  # process right child

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY 


def build_tree(train, max_depth, min_size):
  '''
  Inputs:
    - train: Training set, a list of examples. Each example is a list, whose last element is the label.
    - max_depth: maximum depth of the tree, an integer (root has depth 1)
    - min_size: minimum size of a group, an integer
  Output:
    - root: The root node, a recursive dictionary that should carry the whole tree
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  root=get_best_split(train)
  recursive_split(root,max_depth,min_size,1)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return root

def predict(root, sample):
  '''
  Inputs:
  root: the root node of the tree. a recursive dictionary that carries the whole tree.
  sample: a list
  Outputs:
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  if(sample[root['index']]<root['value']):
    if(isinstance(root['left'],dict)):
        return predict(root['left'],sample)
    else:
        return root['left']
    
  if(sample[root['index']]>=root['value']):
    if(isinstance(root['right'],dict)):
        return predict(root['right'],sample)
    else:
        return root['right']
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
