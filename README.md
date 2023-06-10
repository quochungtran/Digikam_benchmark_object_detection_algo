# benchmark_object_detection_algo

As part of the workflow for object detection in digiKam, I plan to implement an offline analysis The goal is 

Benchmark model performance with a new dataset and new categorical features to see if a pre-trained model can deal with new objects, not be overfitting and bias into new objects, still remains a network ‘s feature to predict all existing objects. With Gilles ' proposing, I need to take into account a dedicated git-fs repository that we create for unit-tests to host collections of files (detail in [2])


Analyzing the components of a deep learning architecture or the different components of the neural network architecture can help identify new object features. How different when we modify the weights of each layer in the network can have a significant impact on the model's performance.

In this part, using Python, this repo is aimed to accelerate the development process and access a wider range of pre-built models, ensuring that the assigned tags are both accurate and relevant to digiKam's use cases.

In this part, I would like to propose some components of model performance in terms of data quality and recommended models that we should care about. 
