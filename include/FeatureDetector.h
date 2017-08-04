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
 * Tie together a feature vector extraction method encapsulated
 * in a FeatureOperator object with a classifier model agnostic
 * to any specific machine learning method.
 * The specified model must of course have been previously trained
 * (or must know how to process) the format of feature vectors
 * provided by the given FeatureOperator object.
 *
 * Richard Palmer
 * 2013
 */

#pragma once
#ifndef RLEARNING_FEATURE_DETECTOR_H
#define RLEARNING_FEATURE_DETECTOR_H

#include "Model.h"
using RLearning::Model;
#include <FeatureOperator.h>
using RFeatures::FeatureOperator;


namespace RLearning
{

class FeatureDetector
{
public:
    FeatureDetector( Model *model, const FeatureOperator *fo);

    cv::Size2f getActualSize() const; // Actual size of the feature in metres

    double detect( const cv::Rect &rct);

private:
    Model *model_;  // Trained model
    const FeatureOperator *fop_;  // Feature operator
};  // end class

}   // end namespace

#endif
