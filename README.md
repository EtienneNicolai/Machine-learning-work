This project explores age prediction from facial images using deep learning and transfer learning techniques. The objective was to adapt pretrained convolutional neural networks (CNNs) for a regression task that predicts a person’s age as a continuous value.
Two approaches were developed and evaluated:

A baseline CNN trained from scratch
A transfer learning model based on ResNet50

The results demonstrate the effectiveness of transfer learning when working with limited and imbalanced datasets.

The goals of this project were to:
Apply convolutional neural networks to a real-world regression problem
Compare a custom CNN against a pretrained architecture
Evaluate model performance using MAE and MSE
Analyse training behaviour, overfitting, and dataset limitations

The dataset consists of approximately 1,000 facial images organized into folders labelled by age.


Preprocessing Steps: These steps were implemented to improve generalisation and reduce overfitting.
Images resized to 224 × 224
Pixel values normalized to [0, 1]
Train/validation split (70/30 and 75/25 tested)
Data augmentation:
Horizontal flipping
Rotation
Zoom
Shifting

Model 1 – Baseline CNN, Custom CNN inspired by MNIST architecture
Trained from scratch
Smaller input resolution (28 × 28 grayscale)
Regression output layer with linear activation
This model served as a performance baseline.
Model 2 – Transfer Learning (ResNet50)
The second model used ResNet50 pretrained on ImageNet as a feature extractor.
Modifications included:
Removal of classification head
Global Average Pooling layer
Dense layer (ReLU activation)
Final Dense(1) output with linear activation for regression
Base model initially frozen
This approach leveraged pretrained visual features to improve age prediction performance.


Training Configuration

Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Metric: Mean Absolute Error (MAE)
Batch Size: 32
Early Stopping enabled
Learning rate reduction on plateau

Results
Model 1 (Baseline CNN)
Tst MAE: ~17.8 years
Test MSE: ~434.6

Predictions tended to cluster around the dataset mean, showing limited generalisation and bias toward average age values.

Model 2 (ResNet50 Transfer Learning)
Test MAE: ~5.5 years
Test MSE: ~40.6

The transfer learning model significantly outperformed the baseline CNN, demonstrating that pretrained feature extraction substantially improves regression performance on limited datasets.

Training curves showed stable convergence with moderate generalisation capability.


Model Behaviour & Limitations
Key observations:
Predictions for underrepresented age groups regressed toward the mean
Dataset imbalance affected performance at age extremes
Interpretability tools (e.g., Grad-CAM) were not implemented
Small dataset size limited robustness

While transfer learning improved performance, dataset diversity remains the largest limiting factor.

Key Insights

Transfer learning dramatically reduces training time and improves performance in low-data scenarios
Regression tasks require different evaluation metrics than classification
Dataset imbalance can introduce prediction bias
Freezing pretrained layers limits domain adaptation
This project reinforced the importance of data quality and distribution in deep learning workflows.

Future Improvements

Fine-tune deeper ResNet layers
Implement Grad-CAM for interpretability
Balance dataset across age ranges
Increase dataset size
Experiment with EfficientNet or ensemble approaches
Deploy model as a lightweight web application


## Requirements 

I created a virtual environment using anaconda and jupyter notebooks. with the following imports

tensorflow
pandas
matplotlib
numpy 
scikit-learn 
