# Usage examples:
#    (1a) 10-fold CV (default): mSVM_RFE(dataset='/chb/sheridanlab/scripts/SVM/data/BEIP.tab')
#    (1b) 5-fold CV: mSVM_RFE(dataset='/chb/sheridanlab/scripts/SVM/data/BEIP.tab', folds=5)
#    (1c) Leave-one-out CV: mSVM_RFE(dataset='/chb/sheridanlab/scripts/SVM/data/BEIP.tab', folds="n")
#    (2a) 10-fold CV (default): test_best_features(dataset='/chb/sheridanlab/scripts/SVM/data/BEIP.tab', rankjson='/chb/sheridanlab/scripts/SVM/data/BEIP_average_feature_rankings.json', maxfeatures=20)
#    (2b) 5-fold CV: test_best_features(dataset='/chb/sheridanlab/scripts/SVM/data/BEIP.tab', rankjson='/chb/sheridanlab/scripts/SVM/data/BEIP_average_feature_rankings.json', maxfeatures=20, folds=5)
#    (2c) Leave-one-out CV: test_best_features(dataset='/chb/sheridanlab/scripts/SVM/data/BEIP.tab', rankjson='/chb/sheridanlab/scripts/SVM/data/BEIP_average_feature_rankings.json', maxfeatures=20, folds="n")
# Author: Warren Winter
# Date: 4/30/2013

import Orange
from Orange.classification import svm
from Orange.evaluation import testing, scoring
from Orange.core import MakeRandomIndicesCV, DomainContinuizer
from Orange.data import Table, Domain
import json
import os
import math
import numpy
from collections import defaultdict
from operator import itemgetter


def jssave(filename, data):
    outfile = file(filename, 'w')
    json.dump(data, outfile, indent=4)
    outfile.close()


def jsload(filename):
    infile = file(filename, 'r')
    data = json.load(infile)
    infile.close()
    return data


def multiple_getweights(data, ntimes):
    internal_folds = MakeRandomIndicesCV(data, folds=ntimes)
    internal_folds_output = {}
    for internal_fold in range(1, ntimes+1):
        internal_train_data = data.select(internal_folds, internal_fold, negate=1)
        tuned_learner = svm.SVMLearnerEasy(folds = 4, kernel_type = svm.kernels.Linear, svm_type = svm.SVMLearner.C_SVC)
        weights = svm.get_linear_svm_weights(tuned_learner(internal_train_data), sum=False)
        internal_scores = defaultdict(float)
        for w in weights:
            magnitude=numpy.sqrt(sum([w_attr ** 2 for attr, w_attr in w.items()]))
            for attr, w_attr in w.items():
                internal_scores[attr] += (w_attr / magnitude) ** 2
        internal_folds_output[internal_fold] = internal_scores
    combined_internal_output = defaultdict(list)
    for d in internal_folds_output:
        for k, v in internal_folds_output[d].iteritems():
            combined_internal_output["%s" %k].append(v)
    weightSNR_per_feature = {}
    for feature_and_scores in combined_internal_output.items():
        weightSNR_per_feature[feature_and_scores[0]] = numpy.mean(feature_and_scores[1])/numpy.std(feature_and_scores[1])
    for feature in weightSNR_per_feature:
        if math.isnan(weightSNR_per_feature[feature]):
            weightSNR_per_feature[feature] = 0.0
    weightSNR_per_feature_list = []
    for i in weightSNR_per_feature:
        attr_name = i.split("Orange.feature.Continuous 'N_")[1].split("'")[0]
        weightSNR_per_feature_list += [(attr_name, weightSNR_per_feature[i])]
    weightSNR_per_feature_list.sort(lambda a, b:cmp(a[1], b[1]))
    return weightSNR_per_feature_list


def mSVM_RFE(dataset, folds=10):
    data = Table(dataset)
    if folds == "n":
        folds = len(data)
    external_folds = MakeRandomIndicesCV(data, folds=folds)
    external_folds_output = {}
    for external_fold in range(1, folds+1):
        train_data = data.select(external_folds, external_fold, negate=1)
        attrs = train_data.domain.attributes
        reduced_train_data = train_data
        iter = 1
        attrScores = {}
        while len(attrs) > 0:
            scores = multiple_getweights(reduced_train_data, ntimes=10)
            numToRemove = 1
            for attr, s in scores[:numToRemove]:
                attrScores[attr] = len(attrScores) 
            attrs = [attr for attr, s in scores[numToRemove:]]
            if attrs:
                reduced_train_data = train_data.select(attrs + [train_data.domain.classVar])
            iter += 1
        external_folds_output[external_fold] = attrScores
    combined_external_output = defaultdict(list)
    for d in external_folds_output:
        for k, v in external_folds_output[d].iteritems():
            combined_external_output["%s" %k].append(v)
    features_average_ranks = {}
    for features_and_scores in combined_external_output.items():
        features_average_ranks[features_and_scores[0]] = numpy.mean(features_and_scores[1])
    ranked_features = sorted(features_average_ranks.items(), key=itemgetter(1), reverse=True)
    jssave(dataset[:-4] + '_average_feature_rankings.json', ranked_features)


def test_best_features(dataset, rankjson, maxfeatures, folds=10):
    data = Table(dataset)
    if folds == "n":
        folds = len(data)
    results_by_featureset = {}
    for num_selected in range(1, maxfeatures+1):
        features_average_ranks_sorted = [str(f[0]) for f in jsload(rankjson)]
        selected_features = features_average_ranks_sorted[:num_selected]
        data_subset = data.select(selected_features + [data.domain.classVar])
        learner = svm.SVMLearnerEasy(svm_type=svm.SVMLearner.C_SVC, folds=4, verbose=0)
        results = testing.cross_validation([learner], data_subset, folds=folds)
        results_by_featureset[num_selected] = ['CA: ', scoring.CA(results, report_se=True)[0],
                            'Brier: ', scoring.Brier_score(results, report_se=True)[0],
                            'AUC: ', scoring.AUC(results)[0],
                            'IS: ', scoring.IS(results)[0],
                            'Sensitivity: ', scoring.Sensitivity(results)[0],
                            'Specificity: ', scoring.Specificity(results)[0],
                            'Precision: ', scoring.Precision(results)[0],
                            'PPV: ', scoring.PPV(results)[0],
                            'NPV: ', scoring.NPV(results)[0],
                            'Features: ', selected_features]
    ranked_featuresets = sorted(results_by_featureset.items(), key=lambda x: x[1], reverse=True)
    jssave(dataset[:-4] + '_performance_by_featureset.json', ranked_featuresets)
