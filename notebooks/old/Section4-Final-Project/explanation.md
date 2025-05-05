### This is the base XGBoost-base solution


The solution is available on github [here](https://github.com/UCSD-Data-Science/Public-CSE255-2022/tree/master/notebooks/Section4-Final-Project/XGBoostCreate_submission/code)

#### components of the solution.

The solution is based on the following components: 

* KDTrees: a method for comparing distributions over an 8 dimensional
  color space.
  
* XGBoost: a boost-trees method using 100 boostiing iterations of a
  depth 2 tree.
  
* Ensemble: 30 boosted tree classifiers that are used to estimate
  variability.
  
* Scaling: the 30 sets of scores are combined to compute the mean and
  std of the scores. The scaled score is computed as
  (raw_score-mean) / std
  
* Prediction: for  non abstaining: compares mean of the scaled scores to zero
  For abstaining: abstain if (std of the scaled scores) > |mean of
  scaled scores|
  
  
### references
* KDTRees: https://en.wikipedia.org/wiki/K-d_tree:
The k-d tree is a binary tree in which every node is a k-dimensional
point. Every non-leaf node can be thought of as implicitly generating
a splitting hyperplane that divides the space into two parts, known as
half-spaces. Points to the left of this hyperplane are represented by
the left subtree of that node and points to the right of the
hyperplane are represented by the right subtree. The hyperplane
direction is chosen in the following way: every node in the tree is
associated with one of the k dimensions, with the hyperplane
perpendicular to that dimension's axis. So, for example, if for a
particular split the "x" axis is chosen, all points in the subtree
with a smaller "x" value than the node will appear in the left subtree
and all points with a larger "x" value will be in the right
subtree. In such a case, the hyperplane would be set by the x value of
the point, and its normal would be the unit x-axis.

* Adaboost: https://en.wikipedia.org/wiki/AdaBoost

AdaBoost, short for Adaptive Boosting, is a statistical classification
meta-algorithm formulated by Yoav Freund and Robert Schapire, who won
the 2003 Gödel Prize for their work. It can be used in conjunction
with many other types of learning algorithms to improve
performance. The output of the other learning algorithms ('weak
learners') is combined into a weighted sum that represents the final
output of the boosted classifier. AdaBoost is adaptive in the sense
that subsequent weak learners are tweaked in favor of those instances
misclassified by previous classifiers. In some problems it can be less
susceptible to the overfitting problem than other learning
algorithms. The individual learners can be weak, but as long as the
performance of each one is slightly better than random guessing, the
final model can be proven to converge to a strong learner.

* XGBoost: https://xgboost.readthedocs.io/en/stable/

XGBoost is an optimized distributed gradient boosting library designed
to be highly efficient, flexible and portable. 

  
  

  
