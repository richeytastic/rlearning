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

#include "NaiveBayesRandomCrossValidator.h"
using RLearning::NaiveBayesRandomCrossValidator;
#include <cassert>


NaiveBayesRandomCrossValidator::NaiveBayesRandomCrossValidator( int tcount, int numIts,
            const cv::Mat &xs, const cv::Mat &labels, int numEVs)
    : RandomCrossValidator( tcount, numIts, xs, labels, numEVs)
{
}   // end ctor


void NaiveBayesRandomCrossValidator::train( const cv::Mat &trainData, const cv::Mat &labels)
{
    assert( trainData.type() == CV_32FC1);
    assert( labels.type() == CV_32SC1);
    assert( trainData.rows == labels.cols);
    CvNormalBayesClassifier *classifier = new CvNormalBayesClassifier( trainData, labels.t());
    model_ = boost::shared_ptr<CvNormalBayesClassifier>( classifier);
}   // end trainModel



float NaiveBayesRandomCrossValidator::validate( const cv::Mat &x)
{
    return model_->predict(x);
}   // end validate

