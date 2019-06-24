# PeerVest

**Recommending Investments Across P2P Lending Marketplaces**
*Alex Shropshire*

### Business Understanding
I am looking to help people augment their investment portfolio by allocating funds to Peer-to-Peer Lending Marketplaces. I have yet to invest my own money in P2P debt instruments yet, and I don’t necessarily trust the credit risk assessments provided off-the-shelf by LendingClub and Prosper Marketplace. I would like to create an application that will enable me to recommend the best loans to invest in each of these marketplaces given a user’s available funds and risk tolerance, gathered from inputs on a desktop app with mobile capability.

### Data Understanding
I will connect to the LendingClub & Prosper Marketplace Investor APIs to gather a large number of loans available for investment. I will use historical data to ensure I am including enough loans that are labeled as either fully paid and defaulted/charged-off. I will explore how to incorporate new listings into my recommendation output as they appear on the site. I will anonymize the loan IDs and other potentially sensitive or data to comply with API guidelines, while citing my sources.

### Data Preparation
My goal is to utilize all completed loan results (for training) and all current loan listings in need of additional funding (for predicting profit and recommending investments). Completed loans are already labeled as “fully paid” or “defaulted/charged off” based on how the repayment process played out. Current/Active loans are labeled as “current” if they have begun paying off the loan, and “still being funded” if they are still looking for additional investment. I'll be connecting to the Investor APIs for each site. Each site also releases comprehensive CSV files should the API connection cause me trouble, which I will need to manually update as new CSVs are released. Each site provides >50 features describing the credit history and characteristics of the anonymous borrower, which may need to be reduced given possible redundancy. Most loans (~85-90%) end up getting paid in full, so there will be also a class imbalance problem to deal with in training

### Modeling
**Model 1 (Probability of Default Filter)**: Classification: Logistic Regression, Decision Tree, Random Forests(sklearn), Gradient Boosting(XGBoost) --- linear programming, Bayesian models, Markov model, neural network, support vector machine, nearest neighbor methods --- Bagging, boosting and stacking
- credit risk models with especially low false negative (type II) errors are desired, since false negative prediction will lead to loss.
- depends not only on the borrower's characteristics but also on the economic environment (https://www.investopedia.com/terms/d/defaultprobability.asp), (https://www.youtube.com/watch?v=KHGGlozsRtA),
https://www.youtube.com/watch?v=KVTj7RIqGsQ)
- All “current” loans will receive a predicted probability of default
- I must decide on a decision threshold that makes sense based on a loan’s probability of default
- Loans that have a probability of default > my threshold will not be passed on to Model #2

**Model 2: (Profit/XIRR Prediction)**: Linear Regression(sklearn,statsmodels), Regression Neural Network/Perceptron (Keras), alternatives?
Of the loans that I predict will NOT default, I will calculate an “effective return” (https://www.lendacademy.com/different-ways-to-calculate-p2p-lending-returns/)


### Evaluation
**Model 1**: Accuracy, Recall, Precision, F1, ROC, AUC, Confusion Matrix (K-S curve?) (https://www.machinelearningplus.com/machine-learning/evaluation-metrics-classification-models-r/)
**Model 2**: R-squared & Root Mean Squared Error (+MSE), plot loss function per epoch

### Deployment
The model will be deployed as a Flask app that can be used to collect a user’s risk tolerance and corresponding predicted return desire, and output a comprehensive list of loans/loan IDs for each platform that the person should craft their P2P portfolio from (https://flask-table.readthedocs.io/en/stable/).
**User Stories**:
- As a user, I want to see a list of recommended loans so that I know which loans to invest in.
- As a user, I want to understand my projected reward given my risk tolerance.
- As a user, I want to know if the app recommendations will make me more money than the off-the-shelf LendingClub/Prosper recommendations.
- As a user, I want all news listings to be featured in the recommendation set so that my investment decision is based on recent updates.
- As a user, I want to be able to filter by loan purpose so that I can curate a mission-driven loan set.
- As a user, I want to visually see how the each set (1. filtered out set, 2. filtered in set, 3. recommendation set, 4. all listed loans) is diversified across loan purposes, LC/Prosper Risk Grades, Payment Period End Dates, Weighted? Average Time Elapsed/Time Left, Weighted Average Probability of Loan Default, Credit Scores, Occupations, Interest Rates, Weighted Average/Effective Interest Rate.
- As a power user/data scientist/developer, I want see a python code output that I can use to execute my own orders based off of the recommendation set.
- As a user, I want to compare how the LendingClub/Prosper Marketplace quotes interest rates over time vs. the federal funds rates over time to see how they correlate and how P2P lending depends on the central bank/greater economy.
