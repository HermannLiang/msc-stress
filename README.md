# Individual research project: quantifying cognitive stress pattern through time, frequency and nonlinear analysis

This is a 9-month individual research project, as a part of my MSc Communication & Signal Processing at Imperial College London. The delieveries of this project includes a 50-page thesis alongside with an [academic poster](https://github.com/HermannLiang/msc-stress/blob/final/misc/poster_copy.pdf)

### Introduction and the experiment

This project quantifies cognitive stress pattern using signal processing and machine learning techniques, in an attempt to verify the famous General Adaption Syndrome (GAS) model. GAS states that our resistance to stress will go through a three-stage process: alarm, resistance and exhaustion. I designed a psychological experiment based on the protocol of the Trier social stress test (TSST). The experiment is divided into 5 stages: rest, preparation for speech, deliever a speech, numerical task and recovery. In the first and the last stage the subject (participant) sit still and breath comfortably.  In the ’Preparation’ stage, the subject has 5 minutes to prepare the speech and subsequently deliever the speech in the next stage. In the fourth stage the subject is required to sequentially subtract 13 from 1347. Each stage lasts for 5 minutes. A diagram of the timeline of the GAS and the TSST can be seen below.

In this project, the physiological behaviours when one perceives cognitive stress are examined by analysing the electrocardiogram (ECG). The ECG data were recorded by an arduino extension device call e-health, stored in .txt file. See `./ECGdatabase` folder.

### Physiological Signal Processing and Feature Extraction

Heart Rate Variability (HRV) were extracted from the ECG recordings, followed by a novel noise filtering scheme and feature extraction.

HRV is essentially the temporal difference between two consecutive Peak of the heart-beat.

Below shows the process of extracting HRV from noisy ECG recordings.
![alt text](https://github.com/HermannLiang/msc-stress/blob/final/misc/ecg_hrv.png "ECG to HRV example")

Signal processing techniques were applied to extract time-domain and frequency-domain features, as well as entropy-based complexity
measures. In total, 47 features were extracted from the ECG signals. The majority of those features are the standard ECG features, however, there also some


### Building Machine Learning Model

Stress state classification using SVM, feedforward NN and LSTM network


### Academic Poster:

