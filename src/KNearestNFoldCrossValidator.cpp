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

#include "KNearestNFoldCrossValidator.h"
using RLearning::KNearestNFoldCrossValidator;


KNearestNFoldCrossValidator::KNearestNFoldCrossValidator( int k, int nf, const cv::Mat &xs, const cv::Mat &labels, int numEVs)
    : NFoldCrossValidator( nf, xs, labels, numEVs), _k(k)
{
    if ( _k < 1) _k = 1;
    if ( _k > 32) _k = 32;
}   // end ctor



void KNearestNFoldCrossValidator::train( const cv::Mat &trainData, const cv::Mat &labels)
{
    _model = boost::shared_ptr<CvKNearest>( new CvKNearest( trainData, labels.t()));
}   // end train



float KNearestNFoldCrossValidator::validate( const cv::Mat &x)
{
    return _model->find_nearest( x, _k);
}   // end validate

