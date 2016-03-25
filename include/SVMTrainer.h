/**
 * SVM to train a model from a bunch of positive and negative images.
 * Implements the Sequential Minimal Optimisation algorithm (by John Platt of Microsoft
 * Research) to efficiently minimise the dual Lagrangian of the constrained SVM optimisation
 * problem. SMO allows the constraint that the sum of an arbitrary pair of Lagrange multiplied
 * class labels and other Lagrange multiplied class labels must be zero to always be satisfied.
 *
 * This version employs heuristics for working set selection per iteration by
 * Keerthi et al. (2001) (first order heuristic)
 * and Fan et al. (2005) (second order heuristic)
 *
 * Richard Palmer
 * May 2012
 */

#pragma once
#ifndef RLearning_SVM_TRAINER
#define RLearning_SVM_TRAINER

#include <vector>
using std::vector;
#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/unordered_set.hpp>
using boost::unordered_set;
#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>

#include "SVMParams.h"
using RLearning::SVMParams;
#include "KernelCache.h"
using RLearning::KernelCache;
#include "SVMClassifier.h"
using RLearning::SVMClassifier;
#include <iostream>
using std::ostream;
using std::istream;
typedef unsigned int uint;


namespace RLearning
{

template <typename T>
class SVMTrainer
{
public:
    static SVMClassifier::Ptr train( const vector<T> &pos, const vector<T> &neg, const SVMParams& svmp);

    // maxThreads default of 0 causes training to use all available cores
    SVMTrainer( const SVMParams &p, uint maxThreads=0) throw (InvalidKernelException);

    // maxThreads default of 0 causes training to use all available cores
    SVMTrainer( const typename KernelFunc<T>::Ptr kernel,
                double cost=1e-1, double convTolerance=1e-3,
                uint maxThreads=0);

    ~SVMTrainer();

    // Optimise the lagrange multiplier weights and return the classifier.
    // The positive and negative example vectors do not have to be the exactly
    // the same length (but should at least be similar in length).
    SVMClassifier::Ptr train( const vector<T> &pos, const vector<T> &neg);

    // Enable or disable error output on the training iterations to show convergence.
    void enableErrorOutput( bool enable);

private:
    uint MAXTHREADS;
    const double COST;            // Cost weighting on misclassified training data
    const double EPS;             // Convergence tolerance (typically 0.001 or 0.0001)
    const typename KernelFunc<T>::Ptr kernel;   // Kernel function (linear, polynomial, gaussian etc)

    KernelCache<T> *kernelCache;  // Kernel cache
    bool enableErrOut_;           // If true, error output (convergence info) displayed
    vector<T> xs;                 // The training instances themselves (negative instances start at negZero)
    vector<double> alphas;        // Lagrange multipliers for each training example
    vector<double> fns;           // Current prediction per training instance
    unordered_set<uint> highIdxs; // High index set
    unordered_set<uint> lowIdxs;  // Low index set
    uint negZero;                 // Zero index to the first negative example in xs, alphas and fns.

    struct Alpha;    // An indexed alpha (Lagrange multiplier)

    void optimise( Alpha &high, Alpha &low, double bDiff);

    // Constrain provided alpha within the allowable region >= 0.0 and <= COST
    double constrainAlpha( double a);

    // Update current functional predictions as well as the first order heuristic
    // for selecting the next pair of alphas to update.
    void updatePredictions( Alpha &high, Alpha &low, double &bHigh, double &bLow);

    // The second order heuristic for selection of the working set partner will normally
    // result in fewer iterations to convergence but at the cost of greater processing time.
    // Original algorithm in "Working Set Selection Using Second Order Information for
    // Training Support Vector Machines", Fan et al., 2005.
    uint selectSecondOrderPartner( const uint i) const;

    // Update membership of the high and low index sets.
    void updateIndexSets( const double alpha, const uint idx);

    // Create and return a new classifier encapsulating the trained weights.
    SVMClassifier::Ptr createClassifier( double threshold) const;

    void reset( const vector<T> &pos, const vector<T> &neg);

    // Return the target value (positive = 1, negative = -1) for the example with given index.
    int target( uint idx) const;

    class ThreadFn; // Function object for multi-threaded updatePredictions()

    static const double TAU;    // Very small positive number
};  // end class SVMTrainer

#include "template/SVMTrainer_template.h"

}   // end namespace

#endif
