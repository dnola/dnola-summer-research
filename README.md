dnola-summer-research
=====================

TITLE: Mayo Clinic and UPenn Seizure Challenge (Ended August 19th) - Machine Learning Project

AUTHOR: David Nola

CHALLENGE PAGE:
https://www.kaggle.com/c/seizure-detection

PROBLEM DESCRIPTION:

Early detection of seizures is vital to a type of therapy through what is called 'responsive electrostimulation'. With a fifteen second lead time, such therapy is often able to abort seizures before they begin in earnest. One way to predict a seizure early is to look at EEG data from a given patient. The Kaggle competition frames the problem in the following way:

Per patient, a series of ictal (seizure and time leading up to seizure) and interictal (non-seizure) EEG segments are collected and broken up into 1 second 'clips'. Each of these clips is assigned a boolean value as to whether the segment is ictal or interictal. If it is an ictal segment, an additional value, latency, is recorded - measuring the time between the start of that clip and an expert-determined start of the seizure. Each clip contains a number of EEG 'channels' taken from different parts of the brain.

The problem then, is to take a set of unlabeled clips, determine whether they signify a future occurence of a seizure, and additionally determine whether each clip is an early indicator of said seizure if a seizure will occur.


OVERVIEW:

Consists of two primary components:

1. Python Codebase
2. Jenkins Job Configurations

The Jenkins job configurations are wrappers around the Python scripts. These jobs serve to manage intermediate data, feature sets, and solution sets - as well as managing the python processes and workload distribution. Additionally, Jenkins maintains a full log of builds, manages version control, and can automatically distribute work to Amazon EC2 nodes.

The Python scripts do the bulk of the work. The primary machine learning module I use is Scikit-Learn - which depends on Numpy and Scipy. The principal object that the main 'Basic.py' script deals with is the 'EEGSegment' - which contains basic metadata about a clip, a dict of features for that clip, and in the case of a labaled piece of data, the latency and ictal/interictal classifications. The general job 'flow' to generate a finished submission is as follows:

1. Initialize the dataset (Data_Pickle_Init in Jenkins)
2. Add features to the data set (Feature_Add_* in Jenkins)
3. For best results, combine features in various ways to allow grid selection to pick an optimal feature combination (Data_Feature_Powerset in Jenkins, for example)
4. If desired, add the Universal Layer (Described Later) - which is a special case feature used to combine the primary per-subject classifier with a pretrained 'universal' classifier (Feature_Add_Universal_Layer).
5. Run the Basic model on the artifact containing the desired features (Train_Regular_All)
6. Build a submission file out of the results of the model (Build_Single_Classifier_Submission)

Doing the above steps trains a model purely to distinguish seizure vs non-seizure. To save time, I relied on a hack in order to determine early vs not early: go through the above steps, but in the initialization step only keep the value of seizure if latency is less than 15s. Then, run the above process again, and combine the results into a single submission using Build_Dual_Classifier_Submission.

The above yields a submittable CSV. My best result used the following features, powersetted, plus the Universal Layer described later: Split Variance, Variance Ratios, FFT components, Channel Sigma thresholds exceeded (Both 1sigma and 2sigma thresholds). The features will be described below.

LEARNING MODEL DESCRIPTION:

UNIVERSAL CLASSIFIER DESCRIPTION:

FEATURES LIST:

CONCLUSION:
