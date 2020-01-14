# TripletNet-on-BCS
Deep Triplet network applied to fault diagnosis of Eletrical Submersible Pumps.

This a research done in the field of diagnosing possible faults in a specific industrial equipment
called Eletrical Submersible pump, which is designed for extracting oil and gas.
Our main objective is to understand what and how deep learning algorithms can learn to extract relevant feature 
that are simultaneously suitable for classification and suitable for visual analysis by an expert.

In this repository, a deep learning approach is combined with traditional machine learning classifiers 
for achieving great classification performance.
Our approach relies on a convolutional architecture trained with a triplet loss function 
for extracting relevant features directly from the raw data.
A previously hand-crafted feature set, created by a specialist over the course of many years of research,
is compared with the newly extracted feature set.

The implemtation uses Pytorch along with Scikit-learn.
