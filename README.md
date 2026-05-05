# Module5-Part1-bitsom_ba_2511666-
Task 1: Key Observations
From the statistical summary, here's what I noticed:

Customer tenure ranges from 1 to 72 months, with an average of about 25 months
Monthly charges vary a lot — from ₹255 to ₹2,156
Average satisfaction score is 6.87 out of 10, which is okay but not great
Most customers pay on time, though some delay up to 31 days
About the target variable (churn):

Only about 1.55% of customers actually churned. This means the dataset is highly imbalanced — almost everyone stays. This is pretty common in telecom data but it creates a problem: if a model just predicts "no churn" for everyone, it would still get 98.45% accuracy while being completely useless.

So for this project, I'll need to focus on metrics like F1-score, Recall, and AUC-ROC instead of just accuracy.

Task 2: Preprocessing Decisions
Why I used One-Hot Encoding (not Label Encoding):

The categorical columns like region, plan_type, etc. don't have any natural order. "North" isn't greater than "South." Label encoding would assign numbers (0, 1, 2...) which might confuse the model into thinking there's a ranking. One-hot encoding avoids this by creating separate binary columns.

I used drop_first=True to avoid multicollinearity — basically one category becomes the reference point.

Why scaling matters:

Monthly charges go up to ₹2,156 while support tickets only go up to 8. Without scaling, the model would give more importance to larger numbers just because they're bigger. StandardScaler brings everything to the same range so the model treats all features fairly.

Why stratified split:

Since only 1.55% of customers churned, a random split could accidentally put most churners in one set. Stratification makes sure both training and testing sets have the same proportion of churned vs retained customers.

Task 3: Model Architecture
I used sklearn's MLPClassifier to build a simple feed-forward neural network.

INPUT (24 features)
       ↓
Hidden Layer 1: 64 neurons, ReLU activation
       ↓
Hidden Layer 2: 32 neurons, ReLU activation
       ↓
OUTPUT: 1 neuron, Sigmoid activation → probability (0 to 1)
The network takes in 24 input features (after encoding), passes them through two hidden layers that learn patterns, and outputs a single probability score for churn.

Task 4: Interpretation of Results
a) Accuracy is misleading here: The model got 97.50% test accuracy which sounds great, but since 98.45% of customers don't churn anyway, even a model that always says "No Churn" would get similar accuracy. So accuracy alone doesn't tell us much.

b) The model is overfitting: Training accuracy was 100% but testing was 97.50%. Training loss was 0.0005 but test loss jumped to 0.192. This big gap means the model memorized the training data rather than learning useful patterns.

c) It couldn't detect churners: Recall for churn class = 0%. The model didn't catch a single churner. It basically learned that predicting "No Churn" every time is the safest bet.

d) AUC-ROC gives some hope: The AUC-ROC of 0.74 shows the model does assign slightly higher probabilities to actual churners — it's not completely random. The problem is the default 0.5 threshold is too high for such imbalanced data.

e) What I'd do differently: To actually detect churners, I'd need to use class weighting, SMOTE oversampling, or lower the decision threshold. The loss did reduce by 99.7% during training, so the network learned — it just learned the wrong thing (majority class pattern).

Task 5: What I Observed from Experiments
Adding more layers (Exp 2): Going from 2 layers to 3 layers (128,64,32) improved AUC from 0.74 to 0.79 and the model actually caught 1 churner. More depth helps capture complex patterns.

Changing activation to Tanh (Exp 3): Using Tanh instead of ReLU with a higher learning rate gave the best AUC (0.87). Tanh outputs values between -1 and 1 which seems to work better for ranking churners, though it still couldn't classify them correctly at 0.5 threshold.

Smaller batch size (Exp 4): Reducing batch from 32 to 16 gave the best results overall — 98.75% accuracy and the only model with a meaningful F1 score (0.2857). Smaller batches create more frequent weight updates which helps the model explore better solutions.

Lower learning rate (Exp 5): Using 0.0001 instead of 0.001 gave the least overfitting — training and testing performance were almost identical. It also had the lowest test loss (0.07). But it needed all 200 epochs because it learns very slowly.

Overall finding: No matter what I changed, all models struggled with the class imbalance. Hyperparameter tuning alone can't fix a fundamental data problem — you need techniques like SMOTE or class weights for that.

My pick: Experiment 4 performed best overall (highest accuracy, only one with non-zero F1 for churn, perfect precision when it did predict churn). For real-world use though, I'd combine Experiment 5's careful learning approach with proper class imbalance handling.
