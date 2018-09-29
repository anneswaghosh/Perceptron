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

  Y = splitted[0]
  X = splitted[1]

  for eg in range (X.shape[0]):
    if (((np.dot (W, X[eg]) + bias) * Y[eg]) < 0):
      W = W + (rate * X[eg] * Y[eg])
      bias = bias + (rate * Y[eg])

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

def compute_best_learning_rate ():
  input_rate = [1, 0.1, 0.01]
  max_mean = 0

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
                               'training0'+ str(cv) +'.data', x, 10)
      f.close()
    
    accuracy_mean = np.mean (accuracy)
    if (accuracy_mean > max_mean):
      max_mean = accuracy_mean

    print ('Mean accuracy for input rate {} = {}'.format (x, accuracy_mean))
  return max_mean

def train_and_dev_test (train_file, dev_file, rate, epochs):
  train_table = create_table (train_file)

  weight_vec = np.random.uniform (-0.01, 0.01, train_table.shape[1] - 1)
  bias = np.random.uniform (-0.01, 0.01)

  best_accuracy = 0

  for train_epoch in range (epochs):
    weight_vec, bias = perceptron (train_table, weight_vec, bias, rate)

  dev_table = create_table (dev_file)

  accuracy = test_perceptron (dev_table, weight_vec, bias, rate)

  if (accuracy > best_accuracy):
    best_accuracy = accuracy
    best_wt_vec = weight_vec
    best_bias = bias

  return best_accuracy, best_wt_vec, best_bias

#Main function
rate = compute_best_learning_rate () 
accuracy, best_wt_vec, best_bias = train_and_dev_test ("dataset/diabetes.train",
                "dataset/diabetes.dev", rate, 20)

test_file = "dataset/diabetes.test"

test_table = create_table (test_file)
test_accuracy = test_perceptron (test_table, best_wt_vec, best_bias, rate)
print (test_accuracy)
