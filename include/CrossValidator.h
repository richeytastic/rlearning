#pragma once
#ifndef RLEARNING_CROSS_VALIDATOR_H
#define RLEARNING_CROSS_VALIDATOR_H

#include <vector>
using std::vector;
#include <iostream>
using std::ostream;
#include <opencv2/opencv.hpp>
#include <boost/foreach.hpp>
#include <boost/unordered_set.hpp>
#include <boost/shared_ptr.hpp>
using boost::unordered_set;
#include "Classification.h" // RLearning
#include "StatsGenerator.h"  // RLearning
#include "PCA.h"    // RLearning
#include <Random.h> // RLIB


namespace RLearning
{

class CrossValidator
{
public:
    typedef boost::shared_ptr<CrossValidator> Ptr;

    // xs: per row example vectors as CV_32FC1
    // labels: CV_32SC1 per column labels (ints for classification)
    // assert( labels.cols == xs.rows)
    // numEVs: Set this cross validator to do PCA on the training sample before each
    // iteration and to use the eigenvectors extracted from these data to project
    // the remaining data into prior to validation. Parameter numEVs gives the number
    // of eigenvectors to use. If set <= 0, no PCA will be done and the original
    // feature vectors will be used. If set to greater than the maximum dimension
    // of the feature vectors, PCA will be done and all the eigenvectors will be
    // used (equivalent in number to the dimensionality of the original feature
    // vectors). By default, no PCA is done in cross validation.
    CrossValidator( const cv::Mat_<float>& xs, const cv::Mat_<int>& labels, int numEVs=0);
    virtual ~CrossValidator(){}

    void processAll();  // Process all iterations in sequence without stopping
    bool next();        // Do the next iteration, returning true while still more to go

    const StatsGenerator* getStatsGenerator() const { return &_rocFinder;}


    // Given a set of positive and negative (2 class) example vectors in xs (as row vectors),
    // split each row out into either posSet or negSet according to the respective class ID in labels.
    // Vectors posSet and negSet are not cleared before use.
    static void splitIntoPositiveAndNegativeClasses( const cv::Mat_<float>& xs, const cv::Mat_<int>& labels,
                                                     vector<cv::Mat_<float> >& posSet,
                                                     vector<cv::Mat_<float> >& negSet);

    // Create the cross validation matrix (xs) that may be used in this class's constructor.
    // Two arrays of row vectors are set as subsequent rows in the returned matrix. This requires
    // that all the provided row vectors have the same size (1 row by N columns).
    // The positive vectors are added after the negative, so the first negRowVectors.size() rows
    // of the returned matrix will contain the negative vectors. Matrix labels is set to a row vector
    // with each column entry either a 1 or a 0 with 1 denoting a positive example and 0 denoting
    // a negative example. Therefore, on return, the first negRowVectors.size() columns of the single
    // row of labels will be 0 with the remaining entries 1.
    static cv::Mat_<float> createCrossValidationMatrix( const vector<cv::Mat_<float> >& negRowVectors,
                                                        const vector<cv::Mat_<float> >& posRowVectors,
                                                        cv::Mat_<int>& labels);


    // A generalised version of the above function which takes row vectors from rowVectors.size() different classes.
    // On return, matrix labels is set with labels in the range [0,rowVectors.size()-1] corresponding to the position
    // in rowVectors. A set of general negative row vectors should be placed in element position 0 of rowVectors to
    // ensure that the label for the negative examples is 0 on return.
    static cv::Mat_<float> createCrossValidationMatrix( const vector< const vector<cv::Mat_<float> >* >& rowVectors,
                                                        cv::Mat_<int>& labels);


    // Get sz samples from vector pop with replacement (allows for multiples of datums)
    static void sampleWithReplacement( const vector<cv::Mat_<float> > &pop,
                                             vector<cv::Mat_<float> > &sampleSet, int sz, rlib::Random&);

    // Get sz samples from vector pop without replacement (ensures unique datums).
    // Returns the indices of the items taken from pop and set in vector sampleSet.
    static unordered_set<int> sampleWithoutReplacement( const vector<cv::Mat_<float> > &pop,
                                                              vector<cv::Mat_<float> > &sampleSet, int sz, rlib::Random&);

protected:
    // Called with new training data for each iteration.
    // trainData:   per row feature descriptors stored as CV_32FC1 vectors
    // trainLabels: per column training labels. Integers for classification, floats for regression.
    // The number of columns of trainLabels must equal the number of rows of trainData.
    virtual void train( const cv::Mat_<float>& trainData, const cv::Mat_<int>& trainLabels) = 0;

    // Called to validate a positive instance using the current model.
    // x: a single row vector of type CV_32FC1
    // Returns the validation result. Non-negative results denote a positive classification.
    virtual float validate( const cv::Mat_<float>& x) = 0;

    // Create the training mask. Length of mask == labs.cols. Training instances marked 1,
    // all other (validation instances) marked 0. Should return the total number of training
    // instances (i.e. the count of 1 elements). Every time this is called, mask is already
    // initialised to a zero'd out array so the implementing child class only has to set
    // the flags for the corresponding training instances.
    virtual int createTrainingMask( const cv::Mat_<int> &labs, const vector<int> &counts, char *mask) = 0;

    // Returns true iff there are more iterations to go (called by next())
    virtual bool moreIterations() const = 0;

    void getClassCounts( vector<int> &ccnts) const;

private:
    int _numEVs;

    cv::Mat_<float> _txs;    // CV_32FC1 (rows == num data) as row vectors (dims == cols)
    cv::Mat_<int> _tlabels;  // CV_32SC1 (cols == num data) (single row)
    vector<int> _cCounts;    // Class counts (index is class, value is num examples)

    ROCFinder _rocFinder;   // Stats
};  // end class

}   // end namespace

#endif
