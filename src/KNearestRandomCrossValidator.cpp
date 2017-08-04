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

#include "KNearestRandomCrossValidator.h"
using RLearning::KNearestRandomCrossValidator;


KNearestRandomCrossValidator::KNearestRandomCrossValidator( int tcount, int numIts,
            const cv::Mat &xs, const cv::Mat &labels, int numEVs)
    : RandomCrossValidator( tcount, numIts, xs, labels, numEVs)
{
}   // end ctor


void KNearestRandomCrossValidator::train( const cv::Mat &trainData, const cv::Mat &labels)
{
    CvKNearest *classifier = new CvKNearest( trainData, labels.t());
    model_ = boost::shared_ptr<CvKNearest>( classifier);
}   // end trainModel



float KNearestRandomCrossValidator::validate( const cv::Mat &x)
{
    return model_->find_nearest(x,1);
}   // end validate

