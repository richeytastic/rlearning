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
