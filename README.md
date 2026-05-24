This is the Project repository for the UZH Module "Advanced Machine Learning".

This is Group 13, and our topic of choice is something about Fake News detection/classification.
The Project Proposal is described in the file:
- PROBLEMSETTING.md

Project:

- '''split_data_script.py''' --> used for train/dev/test split (70/15/15). Created Sets are saved in the '''datasets''' folder.

- '''linguistic_features.py''' returns a cleaned up version of train.csv, test.csv and dev.csv --> train_features.csv, test_features.csv and dev_features.csv respectfully
- 
- '''linguistic_features_aggressive.py''' returns the same 3 csv but cleaned up more agressively, because we noticed that under the origianl linguistic_features.py, the model was learning the wrong things, and it found 'cheatcodes' to classifying real/fake suspiciously well (illustrated in results_1.jpeg); results_2.png shows the results with the more aggresive clean up instead

- '''hybrid_model.py''' --> combines distilBERT and the 4 engineered features through a dynamic gate (uses the cleaned "xxxx_features.csv" datasets); see results_3.png
