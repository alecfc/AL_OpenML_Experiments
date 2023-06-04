# AL_OpenML_Experiments

This repository contains the code used for the master's thesis on Sampling Bias in Active Learning and its Influence on Operational Classification Performance. 
In order to run a configuration of the experimental pipeline please do the following:

1. After cloning the project, please install all packages listed in the requirements.txt file
2. Select the list of datasets used in dataset_list, or create your own list of datasets to use.
3. The ecoli and yeast datasets must be extracted from the Data folder, the code shows how to do this for ecoli.
4. Set the correct experiment name in the call to run_openML_test. The options are Class_Imbalance, AL_Methods, ML_Methods and Initial_Class_Ratio.
5. Set the ML classifier and AL query algorithm.
6. For other changes like the amount of executions and the amount of folds used, change the parameters used in the main run_AL_test function.
7. After running an experiment, the results pkl file will be visible in Results and the generated figures will be in Figures.
