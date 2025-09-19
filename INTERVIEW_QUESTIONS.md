# Advanced Classification Techniques Interview Questions for Credit Risk Analytics

## Mathematical Foundations

1. **Explain the mathematical intuition behind logistic regression as it relates to probability estimation in credit risk modeling. How does the log-odds (logit) transformation specifically help in credit scoring applications?**

2. **In the context of credit risk modeling, explain the mathematical difference between L1 and L2 regularization. When would you prefer one over the other for a credit scorecard development?**

3. **Derive the gradient descent update rule for logistic regression with L1 regularization. How does this change when dealing with sparse features in credit application data?**

4. **Explain the mathematics behind Platt scaling for probability calibration. Why might this be particularly important in credit risk applications where regulatory requirements demand well-calibrated probabilities?**

5. **Discuss the mathematical foundations of isotonic regression for probability calibration. When would you prefer this over Platt scaling in credit risk models?**

## Statistical Understanding

6. **Explain how you would interpret a Kolmogorov-Smirnov (KS) statistic value of 45 versus 25 in the context of a credit scoring model. What are the implications for model discrimination power?**

7. **Describe the relationship between Information Value (IV), Weight of Evidence (WOE), and model performance. Provide the formula for Information Value and explain how you would use it to select variables in a credit scorecard.**

8. **A credit risk model shows a Gini coefficient of 0.65 on your development sample but 0.58 on the validation sample. What statistical tests would you perform to determine if this difference is statistically significant?**

9. **Explain the mathematical relationship between Gini coefficient, AUC-ROC, and the Mann-Whitney U statistic in the context of evaluating credit risk models.**

10. **Derive the formula for Expected Loss (EL) using PD, LGD, and EAD. How does classification model performance directly impact the variance of this estimate?**

## Advanced Classification Techniques

11. **Compare and contrast the mathematical foundations of gradient boosting machines, random forests, and neural networks for credit risk classification. How do their optimization objectives differ?**

12. **Explain how you would implement a cost-sensitive classification approach for credit risk modeling where the cost of misclassifying a defaulter as non-defaulter is 10 times higher than the reverse. Provide specific algorithmic adjustments.**

13. **What is the mathematics behind One-Class SVM, and how would you apply it to fraud detection in lending applications where labeled fraud cases are extremely rare?**

14. **Explain the principles of adversarial validation in the context of ensuring model stability across different lending portfolios. What specific implementation would you use to detect concept drift?**

15. **Describe the mathematical intuition behind XGBoost's handling of missing values and how this might be advantageous in credit applications where missing data is common and potentially informative.**

## Business Problem Translation

16. **A bank wants to develop a new acquisition strategy targeting "thin-file" customers with limited credit history. How would you frame this as a classification problem, and what specific techniques would you employ to handle the sparse data challenge?**

17. **Your current credit risk model has an AUC of 0.75, but the business wants to reduce decline rates while maintaining the same bad rate. Explain how you would modify your classification approach to achieve this business objective.**

18. **A retail bank is concerned about the high rejection rate of "good" customers in certain demographic segments. How would you formulate this as a fairness-aware classification problem while maintaining regulatory compliance?**

19. **Your mortgage default prediction model shows excellent discrimination power (AUC = 0.85) but poor calibration, especially in the highest-risk deciles. How does this translate to real business impact, and what techniques would you employ to address it?**

20. **A credit card issuer wants to implement a dynamic credit line management strategy. How would you translate this into a multi-class classification problem versus a regression problem? What are the trade-offs?**
