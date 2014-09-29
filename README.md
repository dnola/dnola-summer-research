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
