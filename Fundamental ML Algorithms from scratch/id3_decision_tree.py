import numpy as np
 


def total_entropy(X, attribute, label_classes):
    m_total = X.shape[0] # number of examples
    H_total = 0 # initialize total Shannon entropy H

    for c in label_classes:
        #count number of times a given class
        check = X[attribute] == c 
        label_count = X[check].shape[0]
        
        p = (label_count / m_total) #define proporation p of the count of elements in a class to total elements
        H_class = -p * np.log2(p) #calculate Shannon entropy for each class
        H_total =+ H_class #accumulate to find entropy of current set of classes

    return H_total


def subset_entropy(x, attribute, label_classes):
    m_subset = x.shape[0] # number of examples
    H_subset = 0 # initialize subset Shannon entropy H

    for c in label_classes:
        #count number of times a given class
        H_class = 0
        check = x[attribute] == c 
        label_count = x[check].shape[0]
        if label_count != 0:
            p = (label_count / m_subset) #define proporation p of the count of elements in a class to total elements
            H_class = -p * np.log2(p) #calculate Shannon entropy for each class
       
        H_subset =+ H_class #accumulate to find entropy of current set of classes

    return H_subset


def info_gain(feature_name, X, attribute, label_classes):
     H_total = total_entropy(X, attribute, label_classes) #get total entropy
     
     feature_vals = X[feature_name].unique() #define values for each feature name
     m = X.shape[0] # number of examples
     feature_info_gain = 0 # initialize info gain for single feature

     for val in feature_vals:
         check = X[feature_name] == val #condition
         f_count = X[check].shape[0]
         f_entropy = subset_entropy(X[check], attribute, label_classes)
         f_p = f_count / m
         feature_info_gain += f_p * f_entropy
         
     info_gain = H_total - feature_info_gain
     
     return info_gain


def max_info_gain(X, attribute, label_classes):
    feature_list = X.columns.drop(attribute)
    f_info_gains = []

    for feature_name in feature_list:
        f_info_gain = info_gain(feature_name, X, attribute, label_classes)
        f_info_gains = f_info_gains.append(f_info_gain)

    max_info_feature = max(f_info_gains)

    return max_info_feature


#TODO generate sub-tree, create tree, build id3
