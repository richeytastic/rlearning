/************************************************************************
 * Copyright (C) 2017 Richard Palmer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

#pragma once
#ifndef RLEARNING_DISCRETE_NAIVE_BAYES_H
#define RLEARNING_DISCRETE_NAIVE_BAYES_H

#include <map>
#include <opencv2/opencv.hpp>


namespace RLearning
{

class DiscreteNaiveBayes
{
public:
    // Feature distributions are estimated non-parametrically using histograms
    // of discretised feature values. Minimum number of bins is 2.
    // Parameters minVal and maxVal are the minimum and maxmimum respective
    // values used for the binning of example values into the histograms
    // representing the non-parametric distribution of values in the training data
    // (i.e. minVal and maxVal are respectively the smallest and largest values
    // possible in the training data).
    // Leave Laplacian smoothing constant at 1 unless a VERY good reason to change it exists!
    DiscreteNaiveBayes( int numBins, double minVal, double maxVal, int smooth=1);

    // Add a feature vector (as a multi-channel Matrix with double precision values)
    // for class i. True returned iff the feature vector training instance was added
    // successfully (all training instance x's must have same dimensionality).
    bool addTrainingInstance( int i, const cv::Mat &x);

    // If actual probabilities are required (rather than just values than can be used for
    // maximum a posteriori estimation), client must call this function after all training
    // instances have been added in order to produce the prior feature likelihood matrix fPriors_.
    // If new training instances are added afterwards, this function must be called again to ensure
    // that the feature priors matrix is representative of all the training data.
    // If the client never calls this function, the values returned from calcLogPosterior can
    // only be used in comparison with one another for MAP estimation and not as true probabilities.
    void calcNormalisationFeaturePriors();

    // Returns the smoothed discretised conditional probability of P(F=x|C=i).
    // Features are considered independent (naive assumption)!
    double calcNaiveLogLikelihood( const cv::Mat &x, int i);

    // Calculate log(P(F=x)) from all feature vectors received so far.
    // Features are considered independent (naive assumption)!
    double calcNaiveLogFeaturePrior( const cv::Mat &x);

    // Get the log of the posterior probability P(C=i|F=x). Calculation assumes independence between
    // features (naive assumption). If the client called calcNormalisationFeaturePriors after adding
    // all training data, the returned value can be turned into a probability by exponentiating.
    // Otherwise, the returned value can be used for maximum a posteriori (MAP) estimation only.
    double calcLogPosterior( int i, const cv::Mat &x);

    // Convenience function that does maximum a posteriori (MAP) estimation
    // to determine the most likely class given the provided feature vector.
    int calcMAP( const cv::Mat &X);

    // Returns per-class likelihood values (not neccessarily a probability) given a feature vector.
    std::map<int, double> calcClassLikelihoods( const cv::Mat &X);

    // Return the bin (from 0 to nbins-1) that minVal should be binned into given the
    // minimum and maximum values. Returns -1 if v < minVal or v > maxVal.
    static int binValue( int nbins, double minVal, double maxVal, double v);

private:
    int nbins_;             // Number of bins to split feature measurements into
    const int lapSmooth_;   // Laplacian smoothing constant (typically 1)
    double minVal_;         // Max expected value for binning histogram values
    double maxVal_;         // Min expected value for binning histogram values
    int frowDef_;           // Num rows in feature vectors
    int fcolDef_;           // Num cols in feature vectors (actually * with channels)
    bool createdFeaturePriors_;         // True only after calcNormalisationFeaturePriors called and prior

    std::map<int, int> cCounts_;        // Class instance counts
    int cTotals_;                       // Total count of training instances (feature vectors provided)
    // P(C=i) = cPriors_[i]/cTotals_;

    cv::Mat fCounts_;                   // Binned feature counts - only req for normalisation of probs
    cv::Mat fTotals_;                   // For normalisation of histograms into probability distributions
    cv::Mat fPriors_;                   // Probability densities over the features (fCounts_/fTotals)
    // P(F=x) = product_over(j,k,l)(fPriors(j,k*nbins_ + l)) (naive assumption)
                                        // to the next call to addTrainingInstance.

    std::map<int, cv::Mat> fcCounts_;   // Feature counts for distributions - nbins_ channels.
    std::map<int, cv::Mat> fcTotals_;   // Totals of discretised features given a class
                                        // Matrix elements are features, single channel is total
    // P(F=x|C=i) = product_over(j,k,l)(fcCounts_[i](j,k,l) / fcTotals_[i](j,k))    (naive assumption)


    static const int MIN_NUM_BINS;     // The minimum number of bins allowed (1)
    static const int MIN_SMOOTHING;    // The minimum Laplacian smoothing constant (1)

    void addNewClass(int);
    void setupFeaturePriors();
};  // end class

}   // end namespace

#endif
