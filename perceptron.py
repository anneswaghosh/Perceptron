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

def perceptron (train_table, W, bias, W_a, bias_a, rate, margin, avg, aggr,
                dev_set):
  splitted = np.split (train_table, [1], axis = 1)
  count = 0
  Y = splitted[0]
  X = splitted[1]
  randomize = np.arange (X.shape[0])
  np.random.shuffle (randomize)
  X = X[randomize]
  Y = Y[randomize]

  for eg in range (X.shape[0]):
    if (margin > 0):
      if (aggr == 1):
        if (((np.dot (W, X[eg]) + bias) * Y[eg]) <= margin):
          # Aggressive Perceptron
          rate = (margin-(np.dot (W, X[eg]) + bias) * Y[eg])/(np.dot (X[eg], X[eg])+1)
          W = W + (rate * X[eg] * Y[eg])
          bias = bias + (rate * Y[eg])
          if (dev_set == 1):
            count += 1
      elif (aggr == 0):
        #Margin perceptron
        if (((np.dot (W, X[eg]) + bias) * Y[eg]) < margin):
          W = W + (rate * X[eg] * Y[eg])
          bias = bias + (rate * Y[eg])
          if (dev_set == 1):
            count += 1
    else:
      if (((np.dot (W, X[eg]) + bias) * Y[eg]) <= 0):
        # Standard perceptron
        W = W + (rate * X[eg] * Y[eg])
        bias = bias + (rate * Y[eg])
        if (dev_set == 1 and avg == 0):
          count += 1

      if (avg == 1):
        # Average Perceptron
        W_a = W_a + W
        bias_a = bias_a + bias
        if (dev_set == 1):
          count += 1

  return W, bias, W_a, bias_a, count

def test_perceptron (test_table, W, bias):
  splitted = np.split (test_table, [1], axis = 1)

  Y = splitted[0]
  X = splitted[1]
  error = 0

  for eg in range (X.shape[0]):
    if (((np.dot (W, X[eg]) + bias) * Y[eg]) < 0):
      error += 1

  accuracy = 1 - error/Y.shape[0]
  return accuracy

def compute_best_hyper_params (weight_vec, bias, rate_list, margin_list, decay, avg, aggr):
  max_mean = 0
  max_rate = 0
  max_margin = 0

  for margin in margin_list:
    for rate in rate_list:
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
                       'training0'+ str(cv) +'.data', weight_vec, bias, rate,
                       10, margin, decay, avg, aggr, 0)
        f.close()
    
      accuracy_mean = np.mean (accuracy)
      if (accuracy_mean > max_mean):
        max_mean = accuracy_mean
        max_rate = rate 
        max_margin = margin

      print ('Mean accuracy for input rate {} Margin {} = {}%'.format (rate, margin, accuracy_mean * 100))

  print ('Cross-validation accuracy for best learning rate {} best margin {} = {}%'
          .format (max_rate, max_margin, max_mean * 100))
  return max_rate, max_margin

def train_and_dev_test (train_file, dev_file, weight_vec, bias, rate, epochs,
                        margin, decay, avg, aggr, dev_set):
  train_table = create_table (train_file)
  best_accuracy = 0
  total_updates = 0
  best_update = 0
 
  #Init avg values
  W_a = np.zeros (weight_vec.shape[0])
  bias_a = 0

  for train_epoch in range (epochs):
    if (decay == 1):
      #Reduce the learning rate
      rate = rate/(1+train_epoch)

    weight_vec, bias, W_a, bias_a, count = perceptron (train_table, weight_vec, bias,
                                     W_a, bias_a, rate, margin, avg, aggr, dev_set)

    total_updates = total_updates + count

    dev_table = create_table (dev_file)

    if (avg == 1):
      accuracy = test_perceptron (dev_table, W_a, bias_a)
    else:
      accuracy = test_perceptron (dev_table, weight_vec, bias)

    if (dev_set == 1):
      print ('Accuracy for dev set in epoch {} = {}%'.format (train_epoch, accuracy*100))
      #print ('{}, {}'.format (train_epoch, accuracy*100))

    if (accuracy > best_accuracy):
      best_update = count
      best_accuracy = accuracy
      if (avg == 1):
        best_wt_vec = W_a 
        best_bias = bias_a
      else:
        best_wt_vec = weight_vec
        best_bias = bias

  if (dev_set == 1):
    print ('Total number of updates = {}'.format (total_updates))
    print ('Number of updates for the best epoch = {}'.format (best_update))
  return best_accuracy, best_wt_vec, best_bias

def perceptron_master(weight_vec, bias, rate_list, margin_list, decay, avg, aggr):
  #Cross Validation
  rate, margin = compute_best_hyper_params (weight_vec, bias, rate_list, margin_list, decay, avg, aggr)

  #Train and test Dev set
  accuracy, best_wt_vec, best_bias = train_and_dev_test ("dataset/diabetes.train",
                "dataset/diabetes.dev", weight_vec, bias, rate, 20, margin, decay, avg, aggr, 1)
  print ('Best dev set accuracy = {}%'.format (accuracy*100))

  #Test the actual test set
  test_file = "dataset/diabetes.test"
  test_table = create_table (test_file)
  test_accuracy = test_perceptron (test_table, best_wt_vec, best_bias)
  print ('Test data accuracy = {}%'.format (test_accuracy*100))
  return test_accuracy

def count_labels (Y):
  label_dict = {}

  labels = np.unique (Y)
  for x in range (labels.shape[0]):
    label_dict [labels[x]] = 0

  for x in range (Y.shape[0]):
    label_dict[float (Y[x])] += 1

  return label_dict

def compute_majority_baseline_accuracy (filename, max_label):
  table = create_table (filename)
  splitted = np.split (table, [1], axis = 1)

  Y = splitted[0]
  labels = np.unique (Y)
  label_dict = count_labels (Y)
  total = 0

  for x in range (labels.shape[0]):
    total = total + label_dict[labels[x]]
  
  accuracy = label_dict [max_label]/total
  return accuracy
 
def compute_majority_baseline ():
  train_table = create_table ("dataset/diabetes.train")
  splitted = np.split (train_table, [1], axis = 1)

  Y = splitted[0]
  X = splitted[1]

  labels = np.unique (Y)
  label_dict = count_labels (Y)

  for x in range (labels.shape[0]):
    if (label_dict[labels[x]] >= label_dict[labels[x+1]]): 
      max_label_val = label_dict[labels[x]]
      max_label = labels[x]
      #print ('max_label = {}'.format (labels[x]))
      break
    else:
      max_label_val = label_dict[labels[x+1]]
      max_label = labels[x+1]
      #print ('max_label = {}'.format (labels[x+1]))
      break

  dev_accuracy = compute_majority_baseline_accuracy ("dataset/diabetes.dev",
                                                      max_label)
  print ('Dev accuracy = {}%'.format (dev_accuracy * 100))

  test_accuracy = compute_majority_baseline_accuracy ("dataset/diabetes.test",
                                                      max_label)
  print ('Test accuracy = {}%'.format (test_accuracy * 100))

def invoke_all(seed):
  np.random.seed (seed) 
  weight_vec = np.random.uniform (-0.01, 0.01, 19)
  bias = np.random.uniform (-0.01, 0.01)
  rate_list = [1, 0.1, 0.01]
  avg = 0
  aggr = 0
  decay = 0
  accuracy = 0

  # Simple Perceptron
  margin_list = [0]
  print ("************* Simple Perceptron Start ***************")
  print ('-----------------------------------------------------') 
  accuracy = perceptron_master (weight_vec, bias, rate_list, margin_list, decay, avg, aggr)
  print ("-------------- Simple Perceptron End --------------\n" )

  # Decaying Percepton
  print ("************** Decay Perceptron Start **************") 
  print ('-----------------------------------------------------') 
  decay = 1
  accuracy += perceptron_master (weight_vec, bias, rate_list, margin_list, decay, avg, aggr)
  decay = 0
  print ("-------------- Decay Perceptron End --------------\n" )
  
  # Margin Perceptron
  print ("************** Margin Perceptron Start **************") 
  print ('-----------------------------------------------------') 
  margin_list = [1,0.1,0.01]
  accuracy += perceptron_master (weight_vec, bias, rate_list, margin_list, decay, avg, aggr)
  margin_list = [0] 
  print ("-------------- Margin Perceptron End --------------\n" )

  # Avg Perceptron
  print ("************* Average Perceptron Start **************") 
  print ('------------------------------------------------------') 
  avg = 1
  accuracy += perceptron_master (weight_vec, bias, rate_list, margin_list, decay, avg, aggr)
  avg = 0
  print ("-------------- Average Perceptron End --------------\n" )

  # Aggressive Perceptron
  print ("************* Aggressive Perceptron Start ***************") 
  print ('---------------------------------------------------------') 
  margin_list = [1,0.1,0.01]
  rate_list = [1]
  aggr  = 1
  accuracy += perceptron_master (weight_vec, bias, rate_list, margin_list, decay, avg, aggr)
  margin_list = [0] 
  aggr  = 0
  print ("-------------- Aggressive Perceptron End --------------\n" )

  print ("************* Majority Baseline Classifier Start ***************") 
  compute_majority_baseline ()
  print ("-------------- Majority Baseline Classifier End --------------\n" )
  #return (accuracy/5)

#Start of the execution
invoke_all(42)
'''
best_acc = 0
best_seed = 0
for i in range (0, 50):
  cur_acc = invoke_all(i)
  if (cur_acc > best_acc):
    best_acc = cur_acc
    best_seed = i

print ("Best seed ", best_seed, "Best Accuracy : ", best_acc)
'''
