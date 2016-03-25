/**
 * Two class SVM classifier for OpenCV matrices.
 * Richard Palmer
 * May 2012
 */

#pragma once
#ifndef RLEARNING_SVM_CLASSIFIER_H
#define RLEARNING_SVM_CLASSIFIER_H

#include <vector>
using std::vector;
#include <iostream>
using std::istream;
using std::ostream;
using std::cerr;
using std::endl;
#include <sstream>
using std::stringstream;
using std::ostringstream;
#include <string>
using std::getline;
#include "SVMParams.h"
using RLearning::SVMParams;
#include "KernelFunc.h"
using RLearning::KernelFunc;

#include "Classification.h"

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
typedef unsigned int uint;

#include <FeatureUtils.h>   // RFeatures for reading and writing binary cv::Mat objects


namespace RLearning
{

class SVMClassifier : public RLearning::Classifier
{
public:
    typedef boost::shared_ptr<SVMClassifier> Ptr;
    static Ptr create();

    SVMClassifier(){}   // Enable loading from stream

    // Before supplying 'as', ensure each of its elements has been
    // multipled by the correct class value : {-1,1}
    // On destruction the provided alpha and example vectors WILL BE DELETED
    // unless delVecs is set to false on construction.
    SVMClassifier( const SVMParams &svmp,  // The parameters used to train the classifier
            vector<double> *as, vector<cv::Mat_<float> > *xs,   // Weights and support vectors
            double b, bool delVecs=true,   // Threshold and whether vectors should be deleted by object
            uint numPos=0, uint numNeg=0); // Number of positive and negative training examples used (not req.)

    virtual ~SVMClassifier();   // Deletes provided weight and example vectors unless delVecs=false in c'tor.

    // Returned value >= 0 denotes positive class and < 0 denotes negative class.
    virtual float predict( const cv::Mat_<float> &z) const;

    // Get/set the number of positive and negative examples used in training
    uint getNumPos() const { return numPos;}
    uint getNumNeg() const { return numNeg;}

    // Return the size of this model's examples and place the size of each
    // element in the provided out parameter. The total size of the feature
    // vector per example is the multiplication of these three values.
    cv::Size getModelDims( int *channels) const;

    // Return the kernel function used for this classifier.
    KernelFunc<cv::Mat_<float> >::Ptr getKernel() const { return kernel;}

    bool isLinear() const { return svmp.isLinear();}
    // Only valid if kernel function is linear (inner product)!
    cv::Mat_<float> getLinearWeightsImg() const { return linx;}
    double getThreshold() const { return b;}
    uint getNumSVs() const { return numSVs;}

    const SVMParams& getParams() const { return svmp;}

private:
    vector<double> *as;
    vector<cv::Mat_<float> > *xs;
    double b;       // Learned detection threshold
    uint numSVs;    // Length of *as and *xs (number of support vectors)
    cv::Mat_<float> linx;   // Only for linear classifier
    bool delVecs;   // Deletes as and xs on destruction if true (see c'tor)
    uint numPos;    // Number of positive examples used for training (not req.)
    uint numNeg;    // Number of negative examples used for training (not req.)

    SVMParams svmp; // SVM parameters used to train this classifier
    KernelFunc<cv::Mat_<float> >::Ptr kernel;

    void deleteVectors();   // Deletes vectors only if delVecs == true
    void setKernel( const SVMParams&);

    friend ostream& operator<<( ostream &os, const SVMClassifier &svmc);
    friend istream& operator>>( istream &is, SVMClassifier &svmc);
};  // end class SVMClassifier


ostream& operator<<( ostream &os, const SVMClassifier &svmc);
istream& operator>>( istream &is, SVMClassifier &svmc);

}   // end namespace

#endif
