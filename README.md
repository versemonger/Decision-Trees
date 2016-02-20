The dataset comes from [UCL Machine Learning Respository](https://archive.ics.uci.edu/ml/datasets/Mushroom)

Run script allMode will execute the ID3.py in 8 modes and additionaly execute the program on validation data set and generate validation result in the file validation_result.txt. The setting for validation is by default entropy and at a confidence level of 95.
The following are instructions about how to change the setting of ID3.py with optional arguments.
```
-m, --mis_classify_error Use mis_classification error rate as criterion for splitting, info gain method is default method 
-c {0,50,95,99}, --confidence_level {0,50,95,99} Specify the confidence level
-v, --validation      Validate the model with validation data set
-d, --display_tree    Display the decision tree.
```
