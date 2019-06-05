# Individual research project: quantifying cognitive stress pattern through time, frequency and nonlinear analysis

This is a 9-month individual research project, as a part of my MSc Communication & Signal Processing at Imperial College London. The delieveries of this project inclue a 50-page thesis alongside with an [academic poster](https://github.com/HermannLiang/msc-stress/blob/final/misc/poster_copy.pdf). The code was written in MATLAB, apart from the Arduino program was in C++.

## Introduction

This project quantifies cognitive stress pattern using signal processing and machine learning techniques, in an attempt to verify the famous General Adaption Syndrome (GAS) model. In short, I translated this problem into a binary/multi-class classification problem and evaluated different classifiers, namely, support vector machine (SVM), fully-connected neural network and long-short term memory (LSTM) network.

## The Experiment

GAS states that our resistance to stress will go through a three-stage process: alarm, resistance and exhaustion. I designed a psychological experiment based on the protocol of the Trier social stress test (TSST). The experiment is divided into 5 stages: rest, preparation for speech, deliever a speech, numerical task and recovery. In the first and the last stage the subject (participant) sit still and breath comfortably.  In the ’Preparation’ stage, the subject has 5 minutes to prepare the speech and subsequently deliever the speech in the next stage. In the fourth stage the subject is required to sequentially subtract 13 from 1347. Each stage lasts for 5 minutes. A diagram of the timeline of the GAS and the TSST can be seen below.

![alt text](https://github.com/HermannLiang/msc-stress/blob/final/misc/gas_tsst.png "GAS model and the TSST experiment")

In this project, the physiological behaviours when one perceives cognitive stress are examined by analysing the electrocardiogram (ECG). The ECG data were recorded by an arduino extension device call e-health, stored in .txt file. See `./ECGdatabase` folder.

## Physiological Signal Processing and Feature Extraction

Heart Rate Variability (HRV) were extracted from the ECG recordings, followed by noise filtering scheme and feature extraction. HRV is essentially the temporal difference between two consecutive Peak of the heart-beat. 

Below shows the process of extracting HRV from noisy ECG recordings.
![alt text](https://github.com/HermannLiang/msc-stress/blob/final/misc/ecg_hrv.png "ECG to HRV example")

Signal processing techniques were applied to extract time-domain and frequency-domain features, as well as entropy-based complexity measures. In total, 47 features were extracted from the ECG signals. The majority of those features were the standard ECG features, however, some advanced features invented by my colleagues from CSP group at Imperial College London were also used.

![alt text](https://github.com/HermannLiang/msc-stress/blob/final/misc/47_features.jpg "HRV features")

## The Machine Learning Models

**Input**: 47 HRV features, or HRV time-series
**Models**: support vector machine (SVM), fully-connected neural network and long-short term memory (LSTM) network.
**Output**: Categorical labels, according to the stage of the experiment

Three models were employed in this project:

**Support Vector Machine**: For a decade, SVM is the most popular sub-space learning model for solving a classification problem. I trained a RBF kernel SVM with the 47 HRV features listed above, followed by hypterparameter tunning.

**Feedforward neural network** or **fully-connected network** is the simplest type of artificial neural network. In MATLAB, we can import `patternnet` module to build and evaluate the network.

**Long short-term memory (LSTM) network** is widely used in natural language processing, speech
recognition and time-series prediction. It enables us to exploit the temporal information in the HRV features.

The following figure shows the structure of these two neural network.

![alt text](https://github.com/HermannLiang/msc-stress/blob/final/misc/network.PNG "network")

## Evaluation

I applied three dimensionality reduction techniques to increase the training speed and only retain the optimal combination of features that show discrimination among different classes. There are:

* Statistical analysis: Analysis of Variance (ANOVA)
* Sequential feature selection
* Principal component analysis (PCA) 

I also applied two partition methods during the validation stage:

* Leave-one-out: data from one subject allocated to testing set, the rest to the testing set (repeating over 16 times).
* Shuffled: training and testing sets were pooled from the whole data set, irrespective of which subject they belong to.

In addition, I trained the another LSTM network with HRV time-series only, bypassing the feature extraction stage. 

## Conclusion

The following table summarized the classification accuracies of different models.

![alt text](https://github.com/HermannLiang/msc-stress/blob/final/misc/res.PNG "Classification accuracy table")

We observed the following:

* Sequential feature selection scores the highest accuracy, at the cost of computation time
* Dimensionality reduction via PCA is a good compromise between accuracy and efficiency
* LSTM network accurately predict the labels of unseen data.
* Training the LSTM directly with HRV series only achieved 78.73% accuracy.
