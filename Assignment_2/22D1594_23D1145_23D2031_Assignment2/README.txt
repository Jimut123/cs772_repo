README.txt for Assignment - 1, CS-772, DL4NLP
-----------------------------------------

Filename of submission: 22D1594_23D1145_23D2031_Assignment2.zip

Team Members: (From Lowest ID to Highest ID)

Jimut Bahan Pal, 22D1594
Shambhavi Pandey, 23D1145
Saikat Dutta, 23D2031

Please run the bptt_rnn.py to train the recurrent
perceptron using 5-folds, and full training data.
Use report_metrics_kfold_aggregate.py to get the 
mean/average metrics reported in the slides.
Use the report_metrics_kfold.py to get the metrics
in the form of mean +- standard deviation.


We have used Accuracy, Precision, Recall and F1-score 
as metrics to be reported for performance of the model.

Here are the requirements:
--------------------------
numpy == for matrix computations
matplotlib == for visualization
scikit-learn == for metrics
--------------------------

The history folder has all the dumps that 
were required for the models.
You can change the path and load the saved model.

The demo/ folder has the gradio application for the 
final demo using the best full model.




