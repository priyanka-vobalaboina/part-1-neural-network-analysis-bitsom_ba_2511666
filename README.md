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

Task 6: Final Reflection
1. What role do weights and biases play in the model?
Weights are the numerical values that determine how strongly each input feature influences the next layer's neurons. Think of them as "importance scores" — a higher weight means that feature has more influence on the prediction. For example, if the weight connecting payment_delay_days to a hidden neuron is large, it means payment delays strongly affect the churn prediction.

Biases are extra values added to each neuron that allow the model to shift its output. Without biases, every neuron's output would have to pass through the origin (0,0), which limits what the network can learn. Biases give the model flexibility — like adjusting the "baseline" of each neuron.

Together, they form the equation at each neuron:

output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + bias)
In our model, there are 3,713 trainable parameters (3,616 weights + 97 biases) that the network learns during training through backpropagation. Initially they start as random values, and gradually get adjusted to minimize the loss function.

2. Why is an activation function required?
Without activation functions, a neural network would just be a series of linear transformations stacked together — which mathematically simplifies to a single linear equation. No matter how many layers we add, the output would still be a linear combination of inputs. This means the network couldn't learn complex, non-linear patterns like "customers who have high payment delays AND low satisfaction AND short tenure are likely to churn."

Activation functions introduce non-linearity, allowing the network to learn curved decision boundaries and complex relationships.

In our model:

ReLU (hidden layers): f(x) = max(0, x) — passes positive values through and blocks negatives. It's fast, avoids the vanishing gradient problem, and works well in practice.
Sigmoid (output layer): σ(x) = 1/(1+e⁻ˣ) — squashes the output between 0 and 1, which we interpret as the probability of churn.
Without these, our "deep" network would be no better than simple logistic regression.

3. What happens when learning rate is too high or too low?
The learning rate controls how big of a step the optimizer takes when updating weights after each batch.

When learning rate is TOO HIGH (e.g., 0.1 or higher):

The model takes large jumps during optimization
It may overshoot the optimal solution and bounce around
Loss might oscillate wildly or even increase (diverge)
The model fails to converge to a good solution
Like trying to park a car by flooring the accelerator — you'll keep overshooting
When learning rate is TOO LOW (e.g., 0.00001):

The model takes tiny steps, learning very slowly
Training takes much longer (needs many more epochs)
May get stuck in local minima (a "good enough" but not optimal solution)
Might not converge within the allowed epochs
Like walking to a destination one centimeter at a time — you'll get there eventually, but it takes forever
What I observed in my experiments:

Exp 3 (LR = 0.01, higher): Achieved the best AUC-ROC (0.8706) — the faster learning helped explore the loss landscape better
Exp 5 (LR = 0.0001, lower): Needed all 200 epochs to converge, BUT had the least overfitting and lowest test loss (0.0703) — the careful learning generalized better
Exp 1 (LR = 0.001, default): A good middle ground that converged in 81 epochs
The ideal learning rate is a balance — fast enough to converge in reasonable time, but slow enough to find a good minimum.

4. Did your model show signs of underfitting or overfitting? Explain.
My model showed clear signs of OVERFITTING. Here's the evidence:

Indicator,Training,Testing,What it tells us
Accuracy,100.00%,97.50%,Model memorized training data
Loss,0.000479,0.192038,Test loss is ~400x higher!
Churn Detection,Perfect on train,0% recall on test,Didn't generalize

Accuracy	100.00%	97.50%	...
Loss	0.000479	0.192038	...
Churn Detection	Perfect on train	0% recall on test	...
View more
Why this is overfitting (not underfitting):

Overfitting = model learns training data TOO well (including noise), but fails on new data. The huge gap between training performance (perfect) and testing performance (poor on minority class) is the classic sign.

Underfitting would mean the model performs poorly on BOTH training and testing data — that's not our case since training accuracy is 100%.

What caused the overfitting:

Extreme class imbalance (98.45% vs 1.55%) — the model found it "easier" to just memorize that almost everyone doesn't churn
Small dataset (only 2,000 samples with just 31 churners) — not enough churn examples to learn meaningful patterns
No regularization — we didn't use dropout or L2 regularization to prevent memorization
Too many parameters (3,713) relative to the number of minority class samples (25 in training)
How I could fix this in future:

Add Dropout layers (randomly disable neurons during training)
Use L2 regularization (penalize large weights)
Apply SMOTE to create synthetic churn samples
Use class weights to penalize missing churners more heavily
Reduce model complexity (fewer neurons/layers)
Collect more data, especially more churn examples
Summary
This project taught me that building a neural network isn't just about getting high accuracy — it's about understanding what the numbers actually mean. A 97.50% accuracy sounds impressive, but when your model can't detect a single churner, it's practically useless for the business problem. The real learning was understanding how hyperparameters interact, why class imbalance matters, and that evaluation metrics must match the business goal.


