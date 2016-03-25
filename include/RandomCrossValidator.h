/**
 * Cross validation where the client can specify the number of iterations
 * and the number of random examples to use in each iteration. Note that
 * each iteration is independent of the others and so some instances may
 * be used in several iterations.
 *
 * Richard Palmer
 * February 2013
 */

#pragma once
#ifndef RLEARNING_RANDOM_CROSS_VALIDATOR_H
#define RLEARNING_RANDOM_CROSS_VALIDATOR_H

#include "CrossValidator.h" // RLearning


namespace RLearning
{

class RandomCrossValidator : public CrossValidator
{
public:
    // tcount: Number of training instances per iteration
    // numIts: Maximum number of iterations
    RandomCrossValidator( int tcount, int numIts, const cv::Mat &xs, const cv::Mat &labels, int numEVs=0);
    virtual ~RandomCrossValidator(){}

    virtual void printResults( ostream &os) const;

protected:
    virtual int createTrainingMask( const cv::Mat &labs, const vector<int> &counts, char *mask);

    virtual bool moreIterations() const;

private:
    int tcount_;             // Number of training instances (positive & negative for each iteration)
    int maxIts_;             // Maximum iterations
    int iter_;               // Iteration of cross validation
};  // end class

}   // end namespace

#endif
