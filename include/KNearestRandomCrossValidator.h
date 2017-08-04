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
#ifndef RLEARNING_KNEAREST_RANDOM_CROSS_VALIDATOR_H
#define RLEARNING_KNEAREST_RANDOM_CROSS_VALIDATOR_H

#include <boost/shared_ptr.hpp>
#include "RandomCrossValidator.h"
using RLearning::RandomCrossValidator;


namespace RLearning
{

class KNearestRandomCrossValidator : public RandomCrossValidator
{
public:
    KNearestRandomCrossValidator( int tcount, int numIts,
                    const cv::Mat &xs, const cv::Mat &labels, int numEVs=0);
    virtual ~KNearestRandomCrossValidator(){}

protected:
    virtual void train( const cv::Mat &trainData, const cv::Mat &labels);
    virtual float validate( const cv::Mat &x);

private:
    boost::shared_ptr<CvKNearest> model_;
};  // end class


}   // end namespace

#endif
