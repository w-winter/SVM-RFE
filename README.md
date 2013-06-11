SVM-RFE
=======     

Release Date: 4/30/2013       
Author: Warren Winter     

---

This script implements with the Orange machine learning library an algorithm for extracting and ranking features that carry the most discriminative or predictive power for an observation's class membership.  It can be used to improve the performance of classifiers, as well as to aid the discovery of biomarkers.

It is based off of the SVM-RFE algorithm first developed by [Guyon et al., 2002](http://axon.cs.byu.edu/Dan/778/papers/Feature%20Selection/guyon*.pdf), critiqued by [Ambroise et al., 2002](http://www.ncbi.nlm.nih.gov/pubmed/11983868), and refined by [Duan et al., 2005](http://www.ncbi.nlm.nih.gov/pubmed/16220686).

---

In outline:

*  I. Rank the dataset's features
	1.	Divide the data into multiple (e.g., 10) "external" folds
	2.	Within each external fold:
		*  Until all features have been ranked:
			1.	Divide the data into 10 "internal" folds
			2.	Within each internal fold:
				1.	Perform grid search (with 4-fold internal CV) to optimize a linear SVM's C parameter for the internal fold's data
				2.	Train a linear SVM on the internal fold's observations
				3.	Normalize the vector of each feature's coefficients in the model
				4.	Calculate weights for each feature
			3.	For each feature, calculate an SNR measure to stabilize its weight estimations across the internal folds: mean(weights) / SD(weights)
			4.	Record the rank of and eliminate some number of lowest-weighted features from the external fold's set of features
	3.	Calculate the mean of each feature's rank across all external folds
       
--
         
*	II.	Determine the optimal number of highest-ranked features for classification accuracy
	1.	For some number of top-ranked features ranging from 1 - n:
		1.	Constrain the features of the dataset to such top features
		2.	Divide the data into multiple (e.g., 10) folds
		3.	Within each fold:
			1.	Perform grid search (with 4-fold internal CV) to optimize an RBF-kernel SVM's C and Gamma parameters for the fold's data
			2.	Train an SVM with an RBF kernel on the fold's observations
			3.	Test the SVM on the the held-out observations, record performance metrics
		4.	Calculate the across-folds mean of each performance metric of the SVM trained on the constrained feature set
	2.	Find the feature set which trained the SVM to classify best

---

Usage:    

*	I. To run the first, ranking stage, call the mSVM_RFE() function
	*	mSVM_RFE() takes 2 parameters:    
		-	dataset (a string, specifying the path to your [Orange-ready, tab-delimited data file](http://orange.biolab.si/doc//reference/tabdelimited.htm))
		-	folds (either an int, to specify how many folds to divide the data, or the string "n" to specify n-fold or leave-one-out cross-validation)    
	*	mSVM_RFE() saves the output to a JSON file in the same folder as the .tab data file    
     
--
      
*	II. To run the second, performance testing stage, call the test_best_features() function    
	*	test_best_features() takes 4 parameters:     
		-	dataset (string containing path to your data file)    
		-	rankjson (string containing path to the JSON output from mSVM_RFE)    
		-	maxfeatures (int specifying the number of features in the largest top-feature set you want to train the SVM on    
		-	folds (either an int, to specify how many folds to divide the data, or the string "n" to specify n-fold or leave-one-out cross-validation)    
	*	test_best_features() saves the output to another JSON file in the same folder    

---

To come: plotting number of features x accuracy, permutation test to assess significance, and some further documentation.
