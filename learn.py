#!/usr/bin/env python

# Name: learn.py
# Author: Andy Wilkins, andrew.wilkins@csiro.au, +61 7 3327 4497, Queensland Centre for Advanced Technologies, PO Box 883, Kenmore, Qld, 4069, Australia
# Year: 2019
# Software required: python2 software stack, listed below
# Language: python
# Program size: 21kB




import os
import sys
import numpy as np
from optparse import OptionParser, OptionValueError

from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import backend as K
K.set_image_dim_ordering('th') # (channels, rows, cols) in X
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# parse command line
p = OptionParser(usage="""usage: %prog [options] <data> <output_file_basename>
Attempts to learn info about <data>.
<data> is assumed to be a CSV file with a header indicated by "#"
<data> is assumed to have a line of the form
#Duration d NumChannels n
where d and n are integers that define the 'width' and 'height' of the 'images' in <data>


Here we use a CNN to attempt to distinguish a true even from a false-positive event

Eg:
./%prog -v -a -p -- cnn_data.csv out
""")
p.add_option("-v", action="store_true", dest="verbose",  help="Verbose output")
p.add_option("-p", action="store_true", dest="plot",  help="Plot accuracy and loss")
p.add_option("-a", action="store_true", dest="augment", help="Augment the training dataset using shifts, etc, of the data.")
p.add_option("--mask", action="store", default=0.1, type="float", help="Cut off this fraction of <data> from each end.  Eg, if the 'width' of each 'image' is 100, and mask=0.1, then 10 pixels are cut from the beginning and the end of each image.  The purpose of this is so that the images can be shifted to generate more data.  Default=%default")
p.add_option("--min_training", action="store", default=2000, type="int", help="If the number of training images is less than this quantity, generate more by random shifts of the images.  This option is irrelevant if the -a flag is not chosen.  Default=%default")
p.add_option("--epochs", action="store", default=50, type="int", help="Number of epochs to use.  Default=%default")
p.add_option("--permute", action="store", default=0.5, type="float", help="When augmenting the training set (with the -a flag), permute the data from original Joey channels (1,2,3), (8,9), (10, 11), with frequency given by this value, otherwise perform shifts.  Eg, --permute=0.4 means 40% of the augmentations will be a random channel permute, while 60% of the augmentations will be data shifts.  Default=%default")

(opts, args) = p.parse_args()
if len(args) != 2:
   sys.stderr.write("Incorrect number of arguments.  Run with '-h' to get help on this program\n")
   sys.exit(1)
(data_file, output_file) = args

if opts.mask > 0.4:
   sys.stderr.write("Currently you cannot cut more than 40% from each image.\nYou must choose mask less than 0.4\n")
   sys.exit(1)


seed = 7
np.random.seed(seed)

if opts.verbose: sys.stdout.write("Obtaining Duration and NumChannels from " + data_file + "... ");
image_width = None
image_height = None
f = open(data_file, 'r')
for line in f:
   if image_width != None and image_height != None:
      break
   if not line.strip():
      continue
   line = line.strip();
   if not line.startswith("#"):
      continue
   if line.startswith("#Duration"):
      line = line.split()
      image_width = int(line[1])
      image_height = int(line[3])
f.close()
if image_width == None or image_height == None:
   sys.stderr.write("Cannot determine parameters.  Exiting\n")
else:
   if opts.verbose: sys.stdout.write("Done.  (Width = " + str(image_width) + ", Height = " + str(image_height) + ")\n")



if opts.verbose: sys.stdout.write("Reading data file " + data_file + "... ");
data = read_csv(data_file, header=None, comment='#').values
if opts.verbose: sys.stdout.write("Done.\n")

if opts.verbose: sys.stdout.write("Extracting inputs and outputs from data... ");
num_inputs = len(data[0]) - 1
X_orig = data[:, 0:num_inputs]
Y_orig = data[:, num_inputs]
# reshape into the size of the "images"
X_orig = X_orig.reshape(X_orig.shape[0], 1, image_height, image_width)
if opts.verbose: sys.stdout.write("Done.\n")



def create_cnn_model(training_images):
   if opts.verbose: sys.stdout.write("  Defining model... ");
   num_training = len(training_images)
   height = len(training_images[0][0])
   width = len(training_images[0][0][0])

   model = Sequential()

   version = 10 # Other versions removed for clarity.  Older versions had different Conv2D, etc, layers
   if opts.verbose: sys.stdout.write("Using version " + str(version) + ".")
   if version == 10:
      # experimenting with different Conv2D
      # trying to reduce over-fitting compared with version==7
      model.add(Conv2D(20, (height, 10), input_shape = (1, height, width), activation='relu'))
      model.add(Dropout(0.2))
      model.add(Conv2D(10, (1, 30), input_shape = (1, height, width), activation='relu'))
      model.add(Dropout(0.2))
      model.add(Flatten())
      model.add(Dense(int(0.1 * num_training), kernel_initializer='glorot_uniform', activation='relu'))
      model.add(Dropout(0.2))
   model.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
   if opts.verbose: sys.stdout.write("  Done.\n");

   if opts.verbose: sys.stdout.write("  Compiling model... ");
   # stocastic gradient descent
   # with initial learningrate=0.01
   # and subsequent learningrate = (initial learningrate)/(1 + decay * epochs) [I think!]
   # and momentum=0.9
   # using nesterov
   sgd = SGD(lr=0.01, momentum=0.9, decay=0.01/50, nesterov=True)
   model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
   if opts.verbose: sys.stdout.write("  Done.\n");
   return model

def augment(X, Y, num_augmentations, permute_chance):
   # take X and Y, and shift a random choice of the Xs, padding with zeroes, to produce an augmented dataset
   # This returns just the extra data generated, not the concatenation of the original and the extra.
   extra_X = []
   extra_Y = []
   num_unaugmented = len(Y)
   # remember that
   # X[i][j][k][l] is intensity of: the i^th input 'image', the j^th colour (we only have 1 colour), the k^th row (we have 8 rows), and the l^th column (1500 columns)
   img_width = len(X[0][0][0])
   max_shift = int(img_width * opts.mask - 1.5)
   for i in range(int(num_augmentations * (1.0 - permute_chance))):
      x_to_shift = np.random.randint(0, num_unaugmented)
      shift_by = np.random.randint(-max_shift, max_shift + 1)
      shifted = np.roll(X[x_to_shift][0], shift_by, axis=1) # axis=1 means shift intensities of each row
      shifted = np.array([shifted]) # put back 'color'
      extra_X.append(shifted)
      extra_Y.append(Y[x_to_shift])
   for i in range(int(num_augmentations * permute_chance)):
      x_to_permute = np.random.randint(0, num_unaugmented)
      permute_type = np.random.ranf()
      permuted = np.copy(X[x_to_permute][0])
      if permute_type < 0.2:
         # swap rows 0 and 1 (Joey channels 1 and 2)
         permuted[[0, 1]] = permuted[[1, 0]]
      if permute_type < 0.4:
         # swap rows 0 and 2 (Joey channels 1 and 3)
         permuted[[0, 2]] = permuted[[2, 0]]
      if permute_type < 0.6:
         # swap rows 1 and 2 (Joey channels 2 and 3)
         permuted[[1, 2]] = permuted[[2, 1]]
      elif permute_type < 0.8:
         # swap rows 4 and 5 (Joey channels 8 and 9)
         permuted[[4, 5]] = permuted[[5, 4]]
      else:
         # swap rows 6 and 7 (Joey channels 10 and 11)
         permuted[[6, 7]] = permuted[[7, 6]]
      extra_X.append(np.array([permuted])) # put back 'color'
      extra_Y.append(Y[x_to_permute])
   return (np.array(extra_X), np.array(extra_Y))


if opts.verbose: sys.stdout.write("\n\nPERFORMING A STRATIFIED k-FOLD CROSS VALIDATION WITH 10 FOLDS\n")
summary_file = open(output_file + ".txt", 'w')
summary_file.write("PERFORMING A STRATIFIED k-FOLD CROSS VALIDATION WITH 10 FOLDS\n")
summary_file.close()
summary_file_true_false = open(output_file + "_true_false.txt", 'w')
summary_file_true_false.write("PERFORMING A STRATIFIED k-FOLD CROSS VALIDATION WITH 10 FOLDS\n")
summary_file_true_false.close()
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)
fold_num = 0
fig_num = 0
for train, test in kfold.split(X_orig, Y_orig):
   fold_num += 1

   # split off 10% of the training data to act as validation
   X_train, X_validation, Y_train, Y_validation = train_test_split(X_orig[train], Y_orig[train], test_size=0.1, random_state=seed, stratify=Y_orig[train])

   indices_where_Y_train_zero = np.where(Y_train == 0)[0]
   indices_where_Y_train_one = np.where(Y_train == 1)[0]
   if opts.verbose: sys.stdout.write("\n\nFOLD " + str(fold_num) + ".\n  " + str(len(indices_where_Y_train_one)) + " correct triggers in training (out of " + str(len(np.where(Y_orig == 1)[0])) + ")  " + str(len(indices_where_Y_train_zero)) + " false-positives in training (out of " + str(len(np.where(Y_orig == 0)[0])) + ")\n")
   
   if opts.augment:
      # generate the same number of ones and zeroes in the training set
      num_excess = len(indices_where_Y_train_one) - len(indices_where_Y_train_zero)
      if num_excess > 0:
         if opts.verbose: sys.stdout.write("  Too many ones (correct positives).  Generating " + str(num_excess) + " zeroes (false positives)\n")
         X_gen, Y_gen = augment(X_train[indices_where_Y_train_zero], Y_train[indices_where_Y_train_zero], num_excess, 0.0)
      else:
         if opts.verbose: sys.stdout.write("  Too many zeroes (false positives).  Generating " + str(-num_excess) + " ones (correct positives)\n")
         X_gen, Y_gen = augment(X_train[indices_where_Y_train_one], Y_train[indices_where_Y_train_one], -num_excess, 0.0)
      X_train = np.concatenate((X_train, X_gen))
      Y_train = np.concatenate((Y_train, Y_gen))
      if opts.verbose: sys.stdout.write("  Now number of zeroes = " + str(len(np.where(Y_train == 0)[0])) + " and number of ones = " + str(len(np.where(Y_train == 1)[0])) + "\n")

      # ensure there are enough examples in the training set
      if len(Y_train) < opts.min_training:
         if opts.verbose: sys.stdout.write("  Not enough data.  Generating " + str(opts.min_training - len(Y_train)) + " extra 'images' by shifting and swapping\n")
         X_gen, Y_gen = augment(X_train, Y_train, opts.min_training - len(Y_train), opts.permute)
         X_train = np.concatenate((X_train, X_gen))
         Y_train = np.concatenate((Y_train, Y_gen))
         if opts.verbose: sys.stdout.write("  Now number of zeroes = " + str(len(np.where(Y_train == 0)[0])) + " and number of ones = " + str(len(np.where(Y_train == 1)[0])) + "\n")

   # trim the training set by removing from each end
   num_to_remove = int(len(X_train[0][0][0]) * opts.mask - 0.5)
   if opts.verbose: sys.stdout.write("  Trimming " + str(num_to_remove) + " from each end of the training 'images'\n")
   X_train = np.delete(X_train, np.s_[:num_to_remove], axis = 3)
   X_train = np.delete(X_train, np.s_[-num_to_remove:], axis = 3)
   new_image_width = len(X_train[0][0][0])
   if opts.verbose: sys.stdout.write("  Trimming " + str(num_to_remove) + " from each end of the validation 'images'\n")
   X_validation = np.delete(X_validation, np.s_[:num_to_remove], axis = 3)
   X_validation = np.delete(X_validation, np.s_[-num_to_remove:], axis = 3)
   if opts.verbose: sys.stdout.write("  Now the width of the images is " + str(new_image_width) + "\n")
   

   cnn_model = create_cnn_model(X_train)
   if opts.verbose: sys.stdout.write("  Training:\n")
   if True:
      es = EarlyStopping(monitor = 'val_loss', mode='min', patience=10, min_delta=0.0, verbose=1)
      history = cnn_model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=opts.epochs, batch_size=50, verbose=1, callbacks=[es])
   else:
      history = cnn_model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=opts.epochs, batch_size=50, verbose=1)

   if opts.plot:
      fig_num += 1
      plt.figure(fig_num)
      plt.plot(history.history['acc'])
      plt.plot(history.history['val_acc'])
      plt.title('model accuracy.  Fold = ' + str(fold_num))
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'validation'], loc='upper left')
      plt.savefig(output_file + "_accuracy" + str(fold_num) + ".png")
   
      fig_num += 1
      plt.figure(fig_num)
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss.  Fold = ' + str(fold_num))
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'validation'], loc='upper left')
      plt.savefig(output_file + "_loss" + str(fold_num) + ".png")

   # trim the test data 'images' so they are the same size as training data
   X_test = X_orig[test]
   Y_test = Y_orig[test]
   X_test = np.delete(X_test, np.s_[:num_to_remove], axis = 3)
   X_test = np.delete(X_test, np.s_[-num_to_remove:], axis = 3)

   if opts.verbose: sys.stdout.write("  On the withheld data:\n")
   summary_file = open(output_file + ".txt", 'a')
   summary_file.write("Fold" + str(fold_num) + "\n")
   y_predict = cnn_model.predict_classes(X_test)
   num_true_events = 0
   num_predicted_events = 0
   for i in range(len(Y_test)):
      if Y_test[i] == 1:
         num_true_events += 1
      if y_predict[i] == 1:
         num_predicted_events += 1
   if opts.verbose: sys.stdout.write("    Number of true events = " + str(num_true_events) + " (total size of test data = " + str(len(Y_test)) + ").  Number of predicted events = " + str(num_predicted_events) + "\n")
   summary_file.write("    Number of true events = " + str(num_true_events) + " (total size of test data = " + str(len(Y_test)) + ").  Number of predicted events = " + str(num_predicted_events) + "\n")
   
   num_incorrect = 0
   num_correct = 0
   for i in range(len(Y_test)):
      if Y_test[i] == 1:
         if y_predict[i] == 1:
            num_correct += 1
         else:
            num_incorrect += 1
   if opts.verbose: sys.stdout.write("    Fraction of original true events classified as a true event " + str(100 * float(num_correct)/ float(num_correct + num_incorrect)) + "\n")
   summary_file.write("    Fraction of original true events classified as a true event " + str(100 * float(num_correct)/ float(num_correct + num_incorrect)) + "\n")
   summary_file.write("    Fraction of original true events classified as a false event " + str(100 - 100 * float(num_correct)/ float(num_correct + num_incorrect)) + "\n")

   num_incorrect = 0
   num_correct = 0
   for i in range(len(Y_test)):
      if Y_test[i] == 0:
         if y_predict[i] == 0:
            num_correct += 1
         else:
            num_incorrect += 1
   if opts.verbose: sys.stdout.write("    Fraction of false-positive events classified as a 'not an event' " + str(100 * float(num_correct)/ float(num_correct + num_incorrect)) + "\n")
   summary_file.write("    Fraction of false-positive events classified as a 'not an event' " + str(100 * float(num_correct)/ float(num_correct + num_incorrect)) + "\n")
   summary_file.write("    Fraction of false-positive events classified as a true event " + str(100 - 100 * float(num_correct)/ float(num_correct + num_incorrect)) + "\n")

   summary_file.close()

   summary_file_true_false = open(output_file + "_true_false.txt", 'a')
   summary_file_true_false.write("For the images with numbers\n")
   for i in range(len(test)):
      summary_file_true_false.write(str(test[i]) + " ")
   summary_file_true_false.write("\n")
   summary_file_true_false.write("Correct results are\n")
   for i in range(len(Y_test)):
      summary_file_true_false.write(str(int(Y_test[i])) + " ")
   summary_file_true_false.write("\n")
   summary_file_true_false.write("Predictions from fold" + str(fold_num) + " are\n")
   for i in range(len(Y_test)):
      summary_file_true_false.write(str(int(y_predict[i][0])) + " ")
   summary_file_true_false.write("\n")
   summary_file_true_false.close()

summary_file = open(output_file + ".txt", 'r')
tot_num_true = 0
tot_num_pred = 0
frac_true = []
frac_false = []
for line in summary_file:
   if not line.strip():
      continue
   line = line.strip()
   s = line.split("Number of true events = ")
   if len(s) > 1:
      tot_num_true += int(s[1].split()[0])
   s = line.split("Number of predicted events = ")
   if len(s) > 1:
      tot_num_pred += int(s[1].split()[0])
   s = line.split("Fraction of original true events classified as a true event ")
   if len(s) > 1:
      frac_true.append(float(s[1].split()[0]))
   s = line.split("Fraction of false-positive events classified as a 'not an event' ")
   if len(s) > 1:
      frac_false.append(float(s[1].split()[0]))
summary_file.close()
summary_file = open(output_file + ".txt", 'a')
summary_file.write("SUMMARY\n")
summary_file.write("Total number of original true events = " + str(tot_num_true) + "\n")
summary_file.write("Total number of predicted true = " + str(tot_num_pred) + "\n")
summary_file.write("Overall fraction of original true events classified as a true event " + str(np.round(np.mean(frac_true), decimals = 1)) + " +/- " + str(np.round(np.std(frac_true), decimals = 1)) + "\n")
summary_file.write("Overall fraction of original true events classified as a false event " + str(np.round(np.mean([100.0 - ft for ft in frac_true]), decimals = 1)) + " +/- " + str(np.round(np.std([100.0 - ft for ft in frac_true]), decimals = 1)) + "\n")
summary_file.write("Overall fraction of false-positive events classified as 'not an event' " + str(np.round(np.mean(frac_false), decimals = 1)) + " +/- " + str(np.round(np.std(frac_false), decimals = 1)) + "\n")
summary_file.write("Overall fraction of false-positive events classified as an event " + str(np.round(np.mean([100.0 - ff for ff in frac_false]), decimals = 1)) + " +/- " + str(np.round(np.std([100.0 - ff for ff in frac_false]), decimals = 1)) + "\n")
summary_file.close()
sys.exit(0)





sys.exit(0)
