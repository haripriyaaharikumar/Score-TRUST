
run_train.sh  is to compute x_test_mode of CIFAR-10 test dataset for class 0 (set target class from 0 to 9 for all classes)
We use trained CIFAR10 PreactResNet18 model (trained_model/base_model) directory to compute it


run_test.sh is to compute the Trust score (trained_model/trust_model- result of run_trains.sh) for CIFAR-10 dataset for class 0 (set target class from 0 to 9 for all classes)

All necessary files for running run_train.sh and run_test.sh are in te subfolders


TRUST_data_stratification.py gives the value of Table 2 for CIFAR-10 
-- The input csv file is the end result of run_test.sh after running for all classes


###### Requirements

CIFAR10 dataset need to be downloaded 
pytorch 10.2
numpy
pandas
scipy
