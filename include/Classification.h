#pragma once
#ifndef RLEARNING_CLASSIFICATION_H
#define RLEARNING_CLASSIFICATION_H

#include <vector>
using std::vector;
#include <iostream>
using std::ostream;
using std::istream;

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>


namespace RLearning
{

struct Classification
{
    // Print results given true positives, false negatives, true negatives and false positives.
    // Wraps calcMetrics and prints results in a table.
    static void printResultsTable( ostream &os, double tp, double fn, double tn, double fp);

    // Calculate and return the sum of square differences between the two vectors.
    // Size and type of the two matrices must be the same.
    // Euclidean distance can be found by taking the sqrt of this value.
    static double calcSSD( const cv::Mat &v1, const cv::Mat &v2);

    enum Metric
    {
        Precision,      // PPV (Proportion of all predicted positives that actually are positive) = TP/(TP + FP)
        NPR,            // NPR (Proportion of all predicted negatives that actually are negative) = TN/(TN + FN)
        Recall,         // TPR (Proportion of all actual positives that are predicted as such) = TP/(TP + FN)
        Specificity,    // TNR (Proportion of all actual negatives that are predicted as such) = TN/(TN + FP)
        Fallout,        // FPR (Type I Misclassification)  (Proportion of all actual negatives that are predicted as positive) = FP/(FP + TN)
        FNR,            // FNR (Type II Misclassification) (Proportion of all actual positives that are predicted as negative) = FN/(FN + TP) = 1 - TPR
        Accuracy,       // ACC (Proportion of all the data that are correctly classified) = (TP + TN)/(TP + FN + TN + FP)
        F1,             // Harmonic mean of precision and recall = 2TP/(2TP + FP + FN)
        MCC             // Matthews Correlation Coefficient = (TP.TN - FP.FN)/sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
    };  // end enum

    // Calc a metric
    static double calcMetric( double tp, double fn, double tn, double fp, Metric m);
};  // end struct


// Writes data to output stream - one line per datum.
void writeData( ostream &os, const vector<cv::Mat_<float> > &data);
void writeData( ostream &os, const cv::Mat_<float> &rowData);

// Reads data from input stream - one line per datum
// (each cv::Mat is a single row vector of type CV_32FC1)
// Returns true on success.
bool readData( istream &is, vector<cv::Mat_<float> > &data);
bool readData( istream &is, cv::Mat_<float> &rowData);


// Abstract class for two class classifiers.
class Classifier
{
public:
    typedef boost::shared_ptr<Classifier> Ptr;

    // Returned value >= 0 denotes positive class and < 0 denotes negative class.
    virtual float predict( const cv::Mat_<float>&) const = 0;

    // Returns true iff example is from the positive class.
    bool classify( const cv::Mat_<float>& z) const { return predict(z) >= 0 ? true : false;}

protected:
    virtual ~Classifier(){}
};  // end class

}   // end namespace

#endif

