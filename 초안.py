# 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage import measure
from skimage.transform import radon, probabilistic_hough_line
from scipy import interpolate, stats
import tensorflow as tf  # Theano 대신 TensorFlow 사용
from sklearn.model_selection import train_test_split  # cross_validation 대신 최신 모듈 사용
from tensorflow.keras.utils import to_categorical  # np_utils의 to_categorical 대체
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import confusion_matrix
import itertools

#파일 불러오기
import os
print(os.listdir("../input"))
