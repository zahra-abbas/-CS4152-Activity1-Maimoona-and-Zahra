1. Introduction
This activity applied neural networks to regression, binary classification, and multiclass classification. It also examined dropout regularization, model saving and reloading, and training callbacks. The experiments were implemented with TensorFlow/Keras, NumPy, Pandas, Matplotlib, and scikit-learn.
2. Task 1 — NYC Taxi Fare Regression
The regression model used three input features: pickup day of week, pickup hour, and trip distance. The neural network contained two hidden Dense layers with 512 ReLU units each and was trained using Adam with mean absolute error (MAE) as the loss.
Evaluation measure	Observed result
Held-out R²	0.7423
Held-out MAE	$2.2999
Held-out RMSE	$3.5491
Predicted fare: Friday, 5 PM, 2 miles	$10.85
Predicted fare: Saturday, 5 PM, 2 miles	$10.62
Saturday minus Friday prediction	−$0.23
The R² value indicates that the model explained about 74.23% of the variation in held-out fares. The average absolute prediction error was about $2.30. For the two specified trips, the predicted Saturday fare was $0.23 lower than the Friday fare.
2.1 Width and batch-size experiments
The notebook tested widths of 128, 256, 512, and 1024 units with batch sizes 32, 100, and 256, using two runs per setting. The following averages are calculated from the experiment table recorded in the notebook.
Width	Batch size	Mean test MAE ($)	Mean test R²
128	32	2.3649	0.7147
128	100	2.3509	0.7282
128	256	2.3641	0.7231
256	32	2.4344	0.6951
256	100	2.3676	0.7230
256	256	2.3566	0.7178
512	32	2.4301	0.7290
512	100	2.3685	0.7183
512	256	2.3477	0.7232
1024	32	2.3651	0.7265
1024	100	2.3586	0.7239
1024	256	2.3471	0.7293
Among the tested settings, width 1024 with batch size 256 had the lowest mean test MAE (about $2.3471). Width 128 with batch size 100 had the highest mean test R² among these grouped settings (about 0.7282). Differences were modest, so a wider network did not consistently guarantee better performance.
3. Task 2 — Credit Card Fraud Classification
A binary classifier with a 128-unit ReLU hidden layer and sigmoid output was trained to classify legitimate and fraudulent transactions. The dataset was strongly imbalanced: 227,393 legitimate transactions and 417 fraud cases were recorded after cleaning in the notebook (about 0.183% fraud). A stratified split was used to preserve class proportions.
Metric	Unweighted model	Class weights {0:1.0, 1:0.01}	Interpretation
True negatives (TN)	56,845	56,863	Legitimate transactions correctly identified
False positives (FP)	19	1	Legitimate transactions incorrectly flagged
False negatives (FN)	19	83	Fraud cases missed
True positives (TP)	79	15	Fraud cases detected
Fraud recall / TPR	80.61%	15.31%	Higher means more fraud cases detected
False positive rate	0.0334%	0.0018%	Lower means fewer false alarms
Accuracy	99.9333%	99.8525%	Can be misleading on imbalanced data
The unweighted model detected 79 of 98 fraud cases in the test set and missed 19. Applying the requested fraud class weight of 0.01 reduced false positives to one but detected only 15 fraud cases and missed 83. This happens because the specified weight makes fraud errors less important during training. Therefore, the weighted model had lower fraud recall and was less suitable when the goal is to catch fraud. Accuracy alone is not sufficient for evaluating this task.
4. Task 3 — Four-Class Nonlinear Classification
A synthetic nonlinear dataset with four classes was generated, containing 2,000 samples: 1,600 for training and 400 for testing. A neural network with a 128-unit ReLU hidden layer and a four-unit softmax output layer was trained.
Result	Observed value
Test accuracy	25%
Class 0 recall	42%
Class 1 recall	29%
Class 2 recall	10%
Class 3 recall	20%
Predicted class for point (0.2, 0.8)	Class 1
Probabilities for classes 0–3	0.1731, 0.3394, 0.2698, 0.2177
The model achieved 25% test accuracy, which is the same as chance-level accuracy for four balanced classes. The confusion matrix shows that many samples were assigned to the wrong class. The model therefore did not learn the nonlinear class boundaries effectively in this run, and this result should be reported rather than described as successful classification.
5. Task 4 — Dropout Regularization
Dropout rate	Training accuracy	Best validation accuracy	Test accuracy	Train–validation gap
0.1	33.13%	30.94%	25.75%	3.44 percentage points
0.2	32.73%	30.94%	24.75%	3.36 percentage points
0.4	30.55%	31.25%	25.50%	0.86 percentage points
Dropout rate 0.1 had the highest test accuracy at 25.75%, while rate 0.4 produced the smallest gap between final training and best validation accuracy. However, all three test accuracies were low and close together. The higher dropout rate reduced the train–validation gap but did not clearly improve test accuracy.
6. Task 5 — Model Persistence and Training Callbacks
The taxi fare model was saved in H5 format and TensorFlow SavedModel format. After reloading the H5 model, the maximum absolute difference between original and reloaded predictions was 0.0, confirming that the saved model reproduced the same predictions. The SavedModel signatures were available for inference.
EarlyStopping and ModelCheckpoint were also demonstrated. Training stopped after 68 epochs rather than continuing to 100. The best validation MAE was 2.2749, and the saved checkpoint achieved a test MAE of 2.3073 and test R² of 0.7315.
7. Conclusion
The taxi fare regression model produced useful predictions with held-out R² of 0.7423 and MAE of about $2.30. In fraud detection, the unweighted model was substantially better at detecting fraud than the specified class-weighted model, although the latter produced fewer false alarms. The four-class classifier performed poorly at 25% accuracy, indicating that its architecture or training setup needs further tuning. Dropout changed the training–validation gap only modestly and did not materially improve test accuracy. Finally, saving and reloading the taxi model preserved predictions exactly, and callbacks stopped training after validation performance stopped improving.
