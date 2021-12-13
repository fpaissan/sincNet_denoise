clc; clear all;

load('EOG_all_epochs.mat');
csvwrite(['EOG_' num2str(fs) '.csv'], EOG_all_epochs);

load('EMG_all_epochs.mat');
csvwrite(['EMG_' num2str(fs) '.csv'], EMG_all_epochs);

load('EEG_all_epochs.mat');
csvwrite(['EEG_' num2str(fs) '.csv'], EEG_all_epochs);