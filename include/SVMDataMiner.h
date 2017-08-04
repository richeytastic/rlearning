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
 * Uses SVM to learn using hard negatives.
 * Richard Palmer
 * 2012
 */

#pragma once
#ifndef RLEARNING_HARD_NEGATIVE_LEARNER_H
#define RLEARNING_HARD_NEGATIVE_LEARNER_H

#include <string>
using std::string;

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>

#include "SVMTrainer.h"
#include "SVMClassifier.h"
using RLearning::SVMClassifier;

// RFeatures
#include <FeatureUtils.h>
#include <ProHOG.h>
using RFeatures::ProHOG;
#include <ProHOGTools.h>


namespace RLearning
{

class SVMDataMiner
{
public:
    // Create a new hard negative learning instance.
    //
    // Parameter posExamplesDir should point to the directory containing
    // bounded examples of positive instances (i.e. each positive image
    // contains a single example object and that object is bounded exactly
    // by the image bounds). All positive image instances are loaded upon
    // construction and included as original and flipped (about vertical
    // axis) instances.
    //
    // Parameter negImgDir should point to a directory containing images
    // with no instances of the required object present. Random subwindows
    // of these images will be used during training to define the set of
    // negatives. Only the absolute filenames are recorded on construction,
    // negative instances will be collected randomly.
    SVMDataMiner( const string &posExamplesDir,
                  const string &negImgDir,
                  const cv::Size&,   // Pro-HOG extraction cell dims
                  double cost,       // SVM misclassification cost
                  double eps,        // SVM convergence param
                  int maxIterations=10);// Maximum number of iterations for hard negatives mining

    // Iteratively trains an optimal SVM classifier
    SVMClassifier::Ptr train();

private:
    const cv::Size cellDims_;   // Cell dimensions for Pro-HOG extraction

    // SVM cost and convergence parameters
    const double cost_;
    const double eps_;
    int maxIterations_;   // Maximum number of iterations for hard negatives mining

    vector<cv::Mat> posInstances_;      // Positive ProHOG feature vectors
    vector<cv::Mat> negImgs_;           // Negative images
};  // end class

}   // end namespace

#endif

