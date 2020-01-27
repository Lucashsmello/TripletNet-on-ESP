# TripletNet-on-BCS
Deep Triplet network applied to fault diagnosis of Eletrical Submersible Pumps.

Research done by the computer lab [NINFA](http://ninfa.inf.ufes.br/site/en), situated at the Federal University of Espírito Santo, Brazil.

The research focus on the field of diagnosing possible faults in a specific industrial equipment, named Eletrical Submersible pump, specially designed for extracting oil and gas.
This paper provides more details in this field: [Combining Classifiers with Decision Templates for Automatic Fault Diagnosis of Electrical Submersible Pumps](https://content.iospress.com/articles/integrated-computer-aided-engineering/ica574).
Our main objective is to understand what and how deep learning algorithms can learn to extract relevant feature 
that are simultaneously suitable for classification and suitable for visual analysis by an expert.

In this repository, a deep learning approach is combined with traditional machine learning classifiers 
for achieving great classification performance.
Our approach relies on a convolutional architecture trained with a triplet loss function 
for extracting relevant features directly from the raw data.
A previously hand-crafted feature set, created by a specialist over the course of many years of research,
is compared with the newly extracted feature set.

It is implemented in Python 3.6.8. It uses Pytorch 1.3 along with Scikit-learn 0.22.
