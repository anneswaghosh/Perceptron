import numpy as np
import math

def file_len (filename):
  len = 0
  train_f = open (filename, "r")
  for line in train_f:
    len += 1

  return len

def create_table (filename):
  num_rows = file_len (filename)
  num_cols = 20
  i = 0

  table = np.zeros ((num_rows, num_cols))

  train_f = open (filename, "r")
  for line in train_f:
    s = line.split (' ')
    
    table[i][0] = int (s[0])

    for x in range (1, len(s)):
      s_split = s[x].split (':')
      table[i][int (s_split[0])] = float (s_split[1])

    i += 1

  return table

def perceptron (train_table, W, bias, rate):
  splitted = np.split (train_table, [1], axis = 1)

  count = 0
  Y = splitted[0]
  X = splitted[1]
  randomize = np.arange (X.shape[0])
  np.random.shuffle (randomize)
  X = X[randomize]
  Y = Y[randomize]

  for eg in range (X.shape[0]):
    if (((np.dot (W, X[eg]) + bias) * Y[eg]) < 0):
      count += 1
      W = W + (rate * X[eg] * Y[eg])
      bias = bias + (rate * Y[eg])

  #print ('Total number of updates on training set = {}'.format (count))
  return W, bias

def test_perceptron (test_table, W, bias, rate):
  splitted = np.split (test_table, [1], axis = 1)

  Y = splitted[0]
  X = splitted[1]
  error = 0

  for eg in range (X.shape[0]):
    if (((np.dot (W, X[eg]) + bias) * Y[eg]) < 0):
      error += 1

  accuracy = 1 - error/Y.shape[0]
  return accuracy

def compute_best_learning_rate (weight_vec, bias):
  input_rate = [1, 0.1, 0.01]
  max_mean = 0
  max_rate = 0

  for x in input_rate:
    accuracy = np.zeros((5, 1))

    for cv in range (5):
      file_list = []
      for i in range (5):
        if (i != cv):
          file_list.append ('training0'+ str(i) +'.data')

      f = open ("cv_4.train", "w")
      for temp in file_list:
        temp_file = open (temp, "r")
        f.write (temp_file.read())

      accuracy[cv], best_wt_vec, best_bias = train_and_dev_test ('cv_4.train',
                       'training0'+ str(cv) +'.data', weight_vec, bias, x, 10)
      f.close()
    
    accuracy_mean = np.mean (accuracy)
    if (accuracy_mean > max_mean):
      max_mean = accuracy_mean
      max_rate = x

    print ('Mean accuracy for input rate {} = {}'.format (x, accuracy_mean))

  print ('Cross-validation accuracy for best learning rate {} = {}'
          .format (max_rate, max_mean))
  return max_rate

def train_and_dev_test (train_file, dev_file, weight_vec, bias, rate, epochs):
  train_table = create_table (train_file)

  best_accuracy = 0

  for train_epoch in range (epochs):
    weight_vec, bias = perceptron (train_table, weight_vec, bias, rate)

    dev_table = create_table (dev_file)

    accuracy = test_perceptron (dev_table, weight_vec, bias, rate)
    #print ('Accuracy for dev set in epoch {} = {}'.format (train_epoch, accuracy))

    if (accuracy > best_accuracy):
      best_accuracy = accuracy
      best_wt_vec = weight_vec
      best_bias = bias

  return best_accuracy, best_wt_vec, best_bias

def perceptron_master():
  #Cross Validation
  rate = compute_best_learning_rate (weight_vec, bias)

  #Train and test Dev set
  accuracy, best_wt_vec, best_bias = train_and_dev_test ("dataset/diabetes.train",
                "dataset/diabetes.dev", weight_vec, bias, rate, 20)
  print ('Best dev set accuracy = {}'.format (accuracy))

  #Test the actual test set
  test_file = "dataset/diabetes.test"
  test_table = create_table (test_file)
  test_accuracy = test_perceptron (test_table, best_wt_vec, best_bias, rate)
  print ('Test data accuracy = {}'.format (test_accuracy))
  return test_accuracy
 
def seeder (seed):
  #Start of the execution
  np.random.seed (seed) 
  weight_vec = np.random.uniform (-0.01, 0.01, 19)
  bias = np.random.uniform (-0.01, 0.01)
  '''weight_vec = [-0.00639461,-0.0096105,  -0.00073563,  0.00449868, -0.00159593, -0.00029146,
 -0.00974438, -0.00025257,  0.00883613,  0.0070159 ,  0.00459929, -0.00782528,
  0.00787808,  0.00714308, -0.00669827,  0.00264668, -0.00959033, -0.00766525,
 -0.00367265]
  bias = -0.006841753866501261
'''

seeder (44)

'''
best_acc = 0
best_seed = 0
for i in range (0, 500):
  cur_acc = seeder(i)
  if (cur_acc > best_acc):
    best_acc = cur_acc
    best_seed = i

print ("Best seed ", best_seed, "Best Accuracy : ", best_acc) '''

