![Breast_Cancer](https://github.com/rcdeepak/Breast-Cancer-Classification-using-Machine-Learning/assets/118427592/8f3449f6-4983-4795-9b0c-0404b5609690)

# Breast Cancer Classification using Machine Learning
Breast cancer is a common form of cancer that primarily affects women but can also occur in men. Early detection and accurate diagnosis are crucial for effective treatment. This project aims to utilize machine learning algorithms to detect breast cancer with reasonable accuracy. By leveraging a dataset obtained from Kaggle, we explore various ML models to classify breast tumors as either malignant (cancerous) or benign (non-cancerous) based on the provided attributes.


## Objective
The main objective of this project is to develop a machine learning model that can accurately classify breast tumors as malignant or benign based on specific attributes. By training and evaluating various ML algorithms, we aim to identify the model with the highest performance in terms of accuracy, precision, recall, and F1 score. The ultimate goal is to provide a reliable tool for assisting in the early detection and diagnosis of breast cancer, potentially complementing traditional screening methods.
## Dataset Used
The dataset used in this project contains breast cancer data, including the following attributes:

-   ID number
-   Diagnosis (M = malignant, B = benign)
-   Ten real-valued features computed for each cell nucleus:
    -   Radius (mean of distances from center to points on the perimeter)
    -   Texture (standard deviation of gray-scale values)
    -   Perimeter
    -   Area
    -   Smoothness (local variation in radius lengths)
    -   Compactness (perimeter^2 / area - 1.0)
    -   Concavity (severity of concave portions of the contour)
    -   Concave points (number of concave portions of the contour)
    -   Symmetry
    -   Fractal dimension ("coastline approximation" - 1)
## Prerequisites
To run the project code and reproduce the results, the following libraries and dependencies need to be installed:

Python (version 3.9)

pandas 

numpy 

matplotlib 

seaborn 

scikit-learn 

## Methodology
1. Data Importing, Cleaning, and Inspection:
-   Importing the necessary libraries
-   Loading the breast cancer dataset
-   Handling missing values, if any
-   Reshaping the dataset as required
-   Inspecting the dataset for initial analysis
2.  Correlation Analysis:
-   Calculating the correlation between the features
-   Visualizing the correlation using a heatmap
-   Plotting a count plot to analyze the distribution of the diagnosis column (target variable)
3.  Model Creation and Evaluation:
-   Importing various ML algorithms such as Logistic Regression, K-Nearest Neighbors, Support Vector Classifier, Decision Tree Classifier, Random Forest Classifier, AdaBoost Classifier, and Gradient Boosting Classifier
-   Implementing a for loop to fit each model and evaluate its performance based on accuracy, precision, recall, and F1 score
-   Creating a new dataframe to store model names and corresponding evaluation metrics
4.  Model Selection:
-   Selecting the model based on the recall score to minimize false negatives (missed cancer diagnoses)
-   Identifying the top-performing models with higher recall scores (e.g., Gradient Boosting Classifier, SVC, and AdaBoost Classifier)
## Results
By evaluating multiple ML models, we found that the Gradient Boosting Classifier, Support Vector Classifier (SVC), and AdaBoost Classifier demonstrated higher recall scores. These models exhibit better performance in identifying malignant breast tumors, minimizing the risk of false negatives. The selection of the final model can be based on the specific requirements and priorities of the classification task.

It is important to note that the results obtained from these models should be further validated and tested on additional datasets or through cross-validation techniques to ensure their generalizability and robustness.
## Conclusion
This project showcases the application of machine learning algorithms for breast cancer classification. By leveraging a dataset comprising various features extracted from cell nuclei, we successfully trained and evaluated multiple ML models to detect breast cancer
