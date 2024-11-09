# Adult-Income-Prediction-Model
# Introduction
This project delves into the complexities of income inequality by analyzing the Adult Income Dataset from Kaggle (https://www.kaggle.com/datasets/wenruliu/adult-income-dataset). 

The dataset comprises 48,842 records, each representing an individual with 15 attributes:   

•	Demographic: age, gender, race, native-country

•	Socioeconomic: education, education-num, marital-status, relationship

•	Employment: workclass, occupation, hours-per-week

•	Income: income (binary: <=50K or >50K)

# Data Preparation and Cleaning
The dataset was thoroughly examined for missing values and inconsistencies. 
The presence of "?" in the data indicates missing values and this was processed to ensure data quality.

**Statistics for the Data frame Observations:**

•**	Age:** The average age is around 38.6 years, with a standard deviation of 13.7 years. The youngest individual is 17, and the oldest is 90.

•	**fnlwgt:** This column likely represents a weighting factor. Its high values and standard deviation suggest significant variation.

**•	educational-num:** The average number of years of education is around 10.

**•	capital-gain:** This column has a high maximum value, indicating potential outliers or extreme values.

**•	capital-loss:** Similar to capital-gain, it has a high maximum value.

**•	hours-per-week**: The average number of hours worked per week is around 40, with a standard deviation of 12.39.

**Demographic Composition of the Dataset**

workclass: Most individuals are employed in the private sector. 

 education: A significant portion of individuals have a high school diploma. 
 
 marital-status: Married-civil-spouse is the most common marital status. 
 
occupation: Professional-specialty is the most frequent occupation. 

relationship: Husband is the most common relationship status. 

race: White is the most common race. 

native-country: The majority of individuals are from the United States. 

income: More individuals earn less than or equal to $50K per year.

**Analyzing the workclass Column**

The workclass column categorizes individuals based on their employment status. 

The output you provided shows the frequency of each category:

•	**Private: T**he most common category, indicating a large number of individuals employed in private sector jobs.

•	**Self-emp-not-inc:** Self-employed individuals not incorporated.

•	**Local-gov:** Individuals working for local government.

**Analyzing the Occupation Column**

The occupation column categorizes individuals based on their occupation. 

As you've observed, the "Prof-specialty" occupation is the most frequent, followed by "Craft-repair" and "Exec-managerial".

**Analyzing the native-country Column**

The native-country column provides information about the country of origin of individuals. 

As you can see, the majority of individuals in the dataset are from the United States. 

However, there's a significant number of individuals from various other countries, including Mexico, Philippines, Germany, and Canada.

# Age Data Distribution

# Histogram for Age

![image](https://github.com/user-attachments/assets/8ede40eb-3123-4e04-8eb5-b38a52d99d2e)

The histogram reveals a right-skewed distribution, indicating that a larger proportion of individuals in the dataset are younger. 

This is evident from the longer tail on the right side of the distribution. 

The peak of the distribution lies around the 30-40 age range, suggesting that this is the most common age group in the dataset. 

The vertical lines represent the median and mean values. 

The fact that the mean is slightly to the right of the median indicates that the distribution is positively skewed.

# Analyzing the Age Distribution: A Combined Approach

![image](https://github.com/user-attachments/assets/5594bd44-e5e5-4e4a-8a73-90a83d1f8b80)

**Histogram:** 

The histogram confirms the right-skewed nature of the age distribution, indicating a larger proportion of younger individuals in the dataset.

**Box Plot:**

The box plot highlights the presence of outliers, individuals with exceptionally high ages.

The box represents the interquartile range (IQR), which contains 50% of the data. The median (Q2) is visible as a line within the box.

The whiskers extend to 1.5 times the IQR from the quartiles, indicating the range of non-outlier data.

**Key Observations from the Filtered Dataset for Age Column:**

**Age Range:** The filtered data primarily consists of individuals aged 79 and above.
	
**Workclass:** A variety of work classes are represented, including private sector, self-employed, and government jobs.

 **Education:**The education levels vary, with some individuals having higher education (e.g., Masters) and others having lower levels (e.g., 7th-8th grade).

**Marital Status:** A mix of marital statuses is present, including married, divorced, never-married, and widowed.

**Occupation:** A diverse range of occupations is represented, from professional roles to manual labor.
	
**Income: **The majority of individuals in this age group have incomes below $50K, but there are a few exceptions.

**Capital Gains and Losses:** Most individuals in this age group have not experienced significant capital gains or losses.

# Work class Data Analysis

# Count plot for Work class

![image](https://github.com/user-attachments/assets/9c98e704-8493-4022-9e0e-6b083af346ed)

**Key Observations:**

**Dominance of Private Sector:** The "Private" workclass has the highest number of individuals, followed by "Self-emp-not-inc" and "Local-gov".

**Income Disparity**: The distribution of income levels (<=50K and >50K) varies across different workclasses.

**Private Sector:** The majority of individuals in the private sector earn less than or equal to $50K.

**Self-Employed:** Self-employed individuals (both incorporated and not incorporated) 

seem to have a higher proportion of those earning more than $50K compared to other workclasses.

**Government Jobs:** Individuals working in local, state, and federal government jobs have a relatively lower proportion of high earners.

**Without Pay and Never-Worked: **These categories naturally have lower income levels.

# Education and Education Num Data Analysis

**Analysis on Education**

<img width="108" alt="image" src="https://github.com/user-attachments/assets/f8fc924b-2c61-417d-84c6-d16e4fbdb995">

After the analysis, here are the key takeaways:

•	The data suggests a strong correlation between education level and the number of years spent in education.

•	"HS-grad" (High School graduate) is the most common education level, followed by "Some-college".

# Marital Status Data Analysis

**Analysis for Marital Status**

<img width="137" alt="image" src="https://github.com/user-attachments/assets/766ba8e3-8307-4c7a-a9b0-e424d80bc66d">

The marital-status column provides information about the marital status of individuals. 

The most common status is "Married-civ-spouse," indicating a significant number of married individuals in the dataset.

**Key Observations:**

•	**Married Individuals:** A large proportion of individuals are married, either to a civilian spouse or a spouse in the Armed Forces.

•	**Single Individuals**: A significant number of individuals are never-married, divorced, separated, or widowed.

# Occupation Data Analysis

![image](https://github.com/user-attachments/assets/b7ca5a8f-1892-45ad-aee1-afb4c049ca96)

**Dominant Occupations:** The plot reveals that "Prof-specialty" and "Craft-repair" are the most common occupations in the dataset.

****Income Disparity:**** There's a clear distinction in income levels across different occupations:

**Higher-Income Occupations:** Occupations like "Exec-managerial" and "Prof-specialty" have a higher proportion of individuals earning more than $50K.

****Lower-Income Occupations**:** Occupations like "Handlers-cleaners", "Farming-fishing", and "Priv-house-serv" have a higher proportion of individuals earning less than or equal to $50K.

**Mixed Income: **Some occupations, such as "Craft-repair" and "Machine-op-inspct," have a more balanced distribution of income levels.

# Relationship Data Analysis

**Understanding the Distribution:**

<img width="107" alt="image" src="https://github.com/user-attachments/assets/12760163-0731-4b7e-bd80-23e5bbac08a9">

**Key Observations:**
•	Family Structure: The data reflects a diverse range of family structures, including nuclear families, extended families, and single-person households.

•	Dominant Relationship: "Husband" is the most frequent relationship, indicating a significant number of male heads of households.

•	Other Relationships: Categories like "Not-in-family" and "Unmarried" represent individuals who may not be directly related to the head of household.

# Race Data Analysis

**Analyzing the race Column**

![image](https://github.com/user-attachments/assets/f374209d-7613-432c-b706-bb3756ecb3a9)

The race column provides information about the racial identity of individuals. The most common race in the dataset is "White," followed by "Black" and "Asian-Pac-Islander."

•	Racial Diversity: The dataset represents a diverse population with individuals from various racial backgrounds.

•	Dominant Race: The majority of individuals in the dataset are White.

The countplot clearly shows that the majority of individuals in the dataset belong to the White racial group. The other racial groups are significantly less represented.

# Gender Data Distribution

Income distribution across genders shows the count (frequency) of individuals in each income category (<=50K and >50K) for both genders (Female and Male). We can observe that:
•	**More Males in Total: **There are more males (22732 + 9918) than females (14423 + 1769) in the dataset.

•	**Income Disparity: **

o	Females: A higher proportion of females earn less than or equal to $50K (14423) compared to those earning more than $50K (1769).

o	Males: The distribution of income is more balanced among males, with a significant number in both income categories.

# Pie Chart for Gender Analysis

![image](https://github.com/user-attachments/assets/416a7a40-54be-4455-968d-0ff59a0f56a5)

The data suggests a gender gap in income distribution, with a higher proportion of females compared to males falling into the lower income category.

# Capital Gain and Capital Loss Data Analysis

The output shows the distribution of capital gains among individuals. The vast majority (44,807) have no capital gain (0). 

A smaller number of individuals have significant capital gains, with 99,999 being the highest.

**Capital Losses:**
Similarly, the distribution of capital losses shows that most individuals (46,560) have not experienced any capital loss.

A smaller number have incurred losses, with values ranging from 1 to 1977.

**Hours Per Week Data Analysis**

![image](https://github.com/user-attachments/assets/a0da137b-1f51-4ae3-890f-f5a945238ede)

**Boxplot:**

•	Median: The median number of hours worked per week is around 40.

•	Quartiles: The first quartile (Q1) is around 30, and the third quartile (Q3) is around 50.

•	Outliers: There are numerous outliers, indicating individuals who work significantly more or fewer hours than the typical range.

**Histogram:**
•	Peak Around 40: The histogram confirms that the majority of individuals work around 40 hours per week, aligning with the median observed in the boxplot.

•	Right Skew: The distribution is right-skewed, meaning there are a few individuals who work significantly more hours than the average. This is also evident from the 

outliers in the boxplot.

**Analyzing Individuals Working Over 80 Hours per Week**

**From the filtered data, we can observe a few key points about individuals who work over 80 hours per week:**

**Diverse Occupations: **Individuals in various occupations, including self-employed, craft-repair, transport-moving, and professional roles, work long hours.
   
**Income Levels:** While some individuals working over 80 hours per week earn more than $50K, others earn less. This suggests that long hours alone do not guarantee high income.

**Education Levels:** Individuals with different education levels, from high school to doctorate, work long hours.

**Marital Status:** Married individuals are well-represented in this group, indicating that long work hours do not necessarily preclude family life.

# Native Countries Data Analysis

**Key Observations:**

•	Dominance of US Citizens: The vast majority of individuals in the dataset are US citizens, as indicated by the high count for "United-States".

•	Diverse Origins: The dataset includes individuals from a diverse range of countries, highlighting the multicultural nature of the population.

•	Notable Countries: Mexico, Philippines, Germany, and Puerto Rico are among the most frequently represented non-US countries.

![image](https://github.com/user-attachments/assets/8b5efa69-88f8-46c4-802d-8e57825e633a)

• ** Grouping Countries**: You can group similar or neighboring countries into a single category. 

This can be done based on geographic region (e.g., "Central America"), economic development level (e.g., "Developed Countries"), or any other relevant criteria.

• ** Eliminating Countries:** If certain countries have very low counts, you can eliminate them from the plot to focus on the more prominent ones. Here are some approaches:

# Data Transformation

The value_counts() output shows that the dataset is now primarily divided into two categories: "United-States" and "Others." This simplification can be useful for certain 

analyses, especially when focusing on the distinction between US citizens and non-US citizens.

**Boxplots of Numerical Variables**

![image](https://github.com/user-attachments/assets/59efdad1-7b04-4904-a589-cd1af2d352fc)

****Key Observations from the Boxplots:**

**Age:****
•	Median Age: The median age appears to be around 37-38 years.

•	Outliers: There are some outliers on the higher end, indicating individuals who are significantly older than the majority.

**fnlwgt:**
•	Outliers: There are numerous outliers on the higher end, suggesting that the fnlwgt variable has a long right tail.

**Capital-gain:**
•	Outliers: The presence of outliers on the higher end indicates that a few individuals have experienced substantial capital gains.

**Capital-loss:**
•	Outliers: Similar to capital gain, there are outliers on the higher end, suggesting significant capital losses for a few individuals.

**Educational-num:**
•	Distribution: The data is relatively concentrated, with a few outliers on the higher end. This indicates that most individuals have a similar level of education.

**Hours-per-week:**
•	Outliers: There are outliers on both the higher and lower ends, suggesting that some individuals work significantly more or fewer hours than the typical range.

# Correlation Heatmap

![image](https://github.com/user-attachments/assets/3d1cf392-6806-41e5-a7d6-1b54d7cc2998)

•	**Weak Correlations**: Most of the correlations between the numerical variables are weak, as indicated by the values close to 0.

**•	Positive Correlation: **

o	educational-num and capital-gain have a moderate positive correlation, suggesting that individuals with higher education tend to have higher capital gains.

o	educational-num and hours-per-week also show a moderate positive correlation, indicating that individuals with higher education might work more hours.

•**	Negative Correlation: **

o	age and fnlwgt have a weak negative correlation, suggesting that older individuals might have lower final weights.

# Pairplot for the Numerical Variables

![image](https://github.com/user-attachments/assets/211c2eb1-d249-4cb5-b682-c851b234f5b1)

**Key Observations from the Pairplot:**

**1.	Distribution of Numerical Variables:**

**o	Age: **The distribution of age is right-skewed, with a peak around 30-40 years.

o	fnlwgt: This variable appears to have a wide range of values, with some outliers.

o	Educational-num: The distribution is relatively concentrated, with most individuals having a moderate level of education.

o	Capital-gain: The distribution is highly skewed, with a few individuals having very high capital gains.

o	Capital-loss: Similar to capital-gain, the distribution is skewed, with a few individuals having significant capital losses.

o	Hours-per-week: The distribution is roughly bell-shaped, with a peak around 40 hours per week.

**2.	Relationships Between Variables:**

o	Age and Income: There seems to be a weak positive correlation between age and income. Older individuals tend to have higher incomes.

o	Education and Income: There's a positive correlation between education level and income. Higher education levels are associated with higher incomes.

o	Capital Gains/Losses and Income: Individuals with higher capital gains or lower capital losses are more likely to have higher incomes.

o	Hours-per-week and Income: There's a weak positive correlation between hours worked per week and income. 

However, the relationship is not very strong, suggesting that other factors might play a more significant role in determining income.

**3.	Income Disparity:**

o	The color coding based on income level helps visualize the income disparity across different variables. 

For example, individuals with higher capital gains and more education tend to have higher incomes.

# Pie chart of Income

The pie chart visually represents the distribution of income levels in the dataset. 

![image](https://github.com/user-attachments/assets/86100dd6-1d24-447e-b0b5-6f969aafc904)

**Here are the key takeaways:**

Dominance of Lower Income: The majority of individuals in the dataset (76.1%) have an income of less than or equal to $50K per year.

Minority of Higher Income: A smaller proportion (23.9%) of individuals earn more than $50K per year.

# Analyzing the Data Split and Feature Engineering

**Data Split:**

The data is correctly splited into training and testing sets using train_test_split. 

The test_size=0.10 parameter indicates that 10% of the data will be used for testing, while the remaining 90% will be used for training. 

The stratify=y parameter ensures that the class distribution (income levels) is preserved in both the training and testing sets. 

This is important for handling imbalanced datasets.

**Categorical and Numerical Features:**

I have correctly identified the categorical and numerical features in the dataset. 

Categorical features are those that represent categories or groups, while numerical features are those that represent quantities.

# Modelling
# Logistic Regression Model to predict Income Levels

Overall, the code demonstrates a well-defined approach to building and tuning a Logistic Regression model for income level prediction.

It incorporates feature preprocessing, addresses class imbalance, and performs hyperparameter tuning to optimize model performance.

![image](https://github.com/user-attachments/assets/8e9617d7-82d5-4a02-aac1-4600b2d5a8c5)

•	The final step in the pipeline is the Logistic Regression model.

•	This model is used to classify instances into two classes (in this case, whether a person earns more or less than $50K per year).

•	It learns a decision boundary that separates the two classes based on the input features.

**Overall Process:**

Data is fed into the pipeline.
	
Numerical features are standardized.

Categorical features are one-hot encoded.
	
The preprocessed data is fed into the Logistic Regression model.

The model learns to classify instances based on the features.

GridSearchCV explores different hyperparameter combinations for the Logistic Regression model to optimize performance.

The GridSearchCV has determined that a Logistic Regression model with these specific hyperparameter settings performs best on the given dataset. 

This model will likely have a good balance between bias and variance, and it should generalize well to unseen data.

The best_score_ attribute of the GridSearchCV object provides the best cross-validation score achieved by the model with the optimal hyperparameters. In this case, the best

cross-validation accuracy is** 0.8513128650434856.**

This means that, on average, the model correctly predicted the income level for 85.13% of the instances in the cross-validation folds. This is a good performance, 

indicating that the model is able to generalize well to unseen data.

# Analyzing the Model Performance

A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for a specific problem. It is a table with 2

rows and 2 columns that report the number of correct and incorrect predictions.   

**Key Metrics:**

•**	Accuracy:** Overall, how often is the model correct?

•	**Precision**: Of the positive predictions, how many are actually positive?

•	**Recall: **Of all the positive cases, how many did the model correctly identify?

•	**F1-Score:** A harmonic mean of precision and recall.

•	**Support**: The number of samples in each class.

**Interpreting the Results:**

T**est Set:**
•	**Accuracy:** The model is correct 85% of the time.

•	**Precision** for Class 0: 87% of the predictions for class 0 are correct.

•	**Recall for Class 0**: 94% of the actual class 0 instances are correctly predicted.

•	**Precision for Class 1**: 74% of the predictions for class 1 are correct.

•	**Recall for Class 1:** 57% of the actual class 1 instances are correctly predicted.

**Train Set:**

•	**Accuracy: **The model is correct 85% of the time.

•	**Precision for Class 0:** 88% of the predictions for class 0 are correct.

•	R**ecall for Class 0**: 93% of the actual class 0 instances are correctly predicted.

•	**Precision for Class 1:** 73% of the predictions for class 1 are correct.

•	**Recall for Class 1:** 60% of the actual class 1 instances are correctly predicted.

**Observations:**

**•	Model Performance**: The model performs reasonably well on both the training and test sets.

•	**Class Imbalance:** The dataset seems to be imbalanced, with more instances of class 0 than class 1. This might impact the model's performance, especially for the minority class.

**•	Overfitting:** The model's performance on the training set is slightly better than on the test set, which might indicate some overfitting.


![image](https://github.com/user-attachments/assets/7741ce50-b2b4-41e6-875b-82582282b18d)

**Based on the given confusion matrix, we can interpret the results as follows:**

•	**True Positives (TP):** 670 instances were correctly classified as positive (income > 50K).

**•**	True Negatives (TN):**** 3481 instances were correctly classified as negative (income <= 50K).

**•**	False Positives (FP):** **235 instances were incorrectly classified as positive.

**•	False Negatives (FN):** 499 instances were incorrectly classified as negative.


# ROC Curve

![image](https://github.com/user-attachments/assets/715d809c-7412-4c99-bdad-5ad9f7fca643)

The provided ROC curve shows a relatively strong performance for the model. Here's a breakdown of its key features:

**•	AUC (Area Under the Curve):** An AUC of 0.90 indicates that the model has good discriminatory power. A higher AUC generally implies a better model.

**•	True Positive Rate (TPR) or Sensitivity: **This measures the proportion of actual positive cases that are correctly identified by the model.

**•	False Positive Rate (FPR) or Specificity:** This measures the proportion of actual negative cases that are incorrectly identified as positive.

**Key Observations:**

•	The curve is closer to the top-left corner of the plot, indicating a good balance between sensitivity and specificity.

•	A higher AUC suggests that the model is effective in distinguishing between positive and negative classes.

# Precision Curve

![image](https://github.com/user-attachments/assets/a59592f5-2323-4096-bf77-e6e960b23eac)

we have a Precision-Recall curve with an Average Precision (AP) of 0.76. Here's what this means:

**•	Precision:** This measures the proportion of positive predictions that are actually correct. A higher precision indicates fewer false positives.

**•	Recall:** This measures the proportion of actual positive cases that are correctly identified. A higher recall indicates fewer false negatives.

**Key Observations:**

**•	Trade-off between Precision and Recall: **As precision increases, recall tends to decrease, and vice versa. This is a common trade-off in classification models.

**•	AP Score:** The Average Precision (AP) score is a measure of the model's performance across different threshold settings. 

An AP of 0.76 indicates reasonable performance, but there's room for improvement.

# Key Research Questions and Answers

**Question 1:** How is income distributed in the dataset? 

**Answer:** The majority of individuals earn less than or equal to $50K per year.

**Question 2:** How do demographic factors like age, race, and gender relate to income? 

**Answer:** There's a weak positive correlation between age and income. Individuals with higher education levels tend to earn more. Racial and gender disparities in income are also evident.

**Question 3:** How do occupational factors like workclass and occupation relate to income? 

**Answer: **Certain occupations like "Exec-managerial" and "Prof-specialty" are associated with higher income levels. Conversely, occupations like "Handlers-cleaners" and 

"Farming-fishing" have a higher proportion of individuals earning lower incomes.

**Question 4:** How do educational factors like education level and years of education relate to income? 

**Answer:** There's a strong correlation between education level and income. Individuals with higher education levels tend to earn more.

**Question 5:** How do factors like hours worked per week relate to income? 

**Answer: **There's a weak positive correlation between hours worked and income. However, long hours alone don't guarantee high income.

# Recommendation

**Data and Preprocessing Recommendations:**

**1.	Handle Missing Values:** Implement appropriate strategies to handle missing values, such as imputation or removal.

**2.	Feature Engineering:** 

o	Create new features that capture relevant information, such as interaction terms between variables.

o	Consider using techniques like one-hot encoding for categorical variables.

o	Normalize or standardize numerical features to improve model performance.

**3. Address Class Imbalance: 

**Employ techniques like oversampling, undersampling, or class weighting to handle the imbalance in the target variable.

**4.	Data Quality: **Ensure data quality by checking for inconsistencies, outliers, and errors.

**Model Selection and Training Recommendations:**

Experiment with Different Algorithms: Explore other algorithms like XGBoost, Random Forest, or Support Vector Machines to potentially improve performance.
	
Hyperparameter Tuning: Use techniques like Grid Search or Randomized Search to optimize hyperparameters.

Ensemble Methods: Combine multiple models to improve overall performance and reduce overfitting.
	
Regularization: Apply techniques like L1 or L2 regularization to prevent overfitting.
	
**Model Evaluation and Interpretation Recommendations:**

Evaluate Model Performance: Use appropriate metrics like accuracy, precision, recall, F1-score, and ROC-AUC to assess the model's performance.

Visualize Model Performance: Use techniques like confusion matrices, ROC curves, and Precision-Recall curves to visualize the model's performance.

Interpret Model Predictions: Use techniques like SHAP values or partial dependence plots to understand the factors influencing the model's predictions.

# Conclusion

This project provides a valuable exploration of income inequality in the Adult Income Dataset. 

The analysis revealed key insights into the relationships between various factors and income levels. 
However, the model's performance in predicting income requires further improvement. 

By addressing class imbalance, exploring feature engineering, and potentially using different models, you can enhance the model's accuracy and generalizability.
