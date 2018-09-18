Supplementary data for the paper entitled " Stage Call: Cardiovascular Reactivity to Audition Stress in Musicians ". Please see detail of the experiment in the paper. For more information please contact aaron.williamon@rcm.ac.uk
The data are 32 Matlab files (.mat) which are divided into 2 categories: High stress and Low stress. The data are clean HRV extracted from recorded ECG of 16 musicians. The detail of the variables in the data files are shown below:
1)	annotation: annotation(1), annotation(2) and annotation(3), are the length (in minute) of the pre-performance, walking to the stage and the performance.
2)	rr_time_unfilt_clean: the timestamp (in second) of the RR interval associated to the the variable rr_val_unfilt_clean.
3)	rr_val_unfilt_clean: the RR interval (in millisecond) extracted from the R-peaks of recorded ECG.
Note that the variables rr_time_unfilt_clean and rr_val_unfilt_clean are used in the interpolation process (cubic spline, at 8Hz), and the annotation is used to segment the data into pre-performance and the performance.

Gender of subjects referred to the filenames: F = female, M = male
Violinist01: F
Violinist02: M
Violinist03: M
Violinist04: M
Violinist05: M
Violinist06: F
Violinist07: M
Violinist08: F
Violinist09: M
Violinist10: M
Violinist11: F
Flutist01: F
Flutist02: F
Flutist03: M
Flutist04: M
Flutist05: F

