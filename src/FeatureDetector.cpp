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

#include "FeatureDetector.h"
using RLearning::FeatureDetector;
#include <cassert>


FeatureDetector::FeatureDetector( Model *m, const FeatureOperator *fo)
    : model_(m), fop_(fo)
{
    assert( m != NULL);
    assert( fo != NULL);
}   // end ctor


cv::Size2f FeatureDetector::getActualSize() const
{
    return model_->getActualSize();
}   // end getActualSize


double FeatureDetector::detect( const cv::Rect &rct)
{
    const cv::Mat fv = (*fop_)( rct);   // Extract the feature vector
    cv::Mat fv32;
    fv.convertTo(fv32, CV_32F);
    const cv::Mat z = fv32.reshape(1,1);    // Single row CV_32C1
    float v = model_->predict( z);
    return v > 0 ? 1 : 0;
}   // end detect
