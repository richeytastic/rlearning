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
 * Manages the loading and training of a linear SVM classifier using point
 * cloud data given in RFeatures::ViewExtract objects. This class creates
 * a single model and so all training data provided in the given directory
 * should be for the same model.
 *
 * The feature extraction method is currently hard-coded as Pro-HOG.
 *
 * Richard Palmer
 * November 2012
 */

#pragma once
#ifndef RLearning_SVM_VIEW_EXTRACT_TRAINER_H
#define RLearning_SVM_VIEW_EXTRACT_TRAINER_H

#include <string>
using std::string;
#include <list>
using std::list;
#include <vector>
using std::vector;
#include <iostream>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>
#include <opencv2/opencv.hpp>

#include <ViewExtract.h>
using RFeatures::ViewExtract;
#include <ProHOG.h> // RFeatures
#include <RangeGradientsBuilder.h>
#include <ImageGradientsBuilder.h>

#include "SVMTrainer.h"
using RLearning::SVMTrainer;
#include "SVMClassifier.h"
using RLearning::SVMClassifier;
#include "SVMParams.h"
using RLearning::SVMParams;


namespace RLearning
{

class SVMViewExtractTrainer
{
public:
    // Retrieve different types of data from ViewExtract instances
    static cv::Mat_<cv::Vec3b> extractColourImage( const ViewExtract::Ptr);
    static cv::Mat_<float> extractRangeImage( const ViewExtract::Ptr);


    typedef boost::shared_ptr<SVMViewExtractTrainer> Ptr;
    static Ptr create( double cost=1, double eps=1e-4);

    SVMViewExtractTrainer( double cost=1, double eps=1e-4);

    // SVM training parameters - always uses a linear kernel.
    void setCost( double cost);
    void setConvergence( double eps);

    // Load the positive examples returning the number of examples loaded (or negative on error)
    int loadPositives( const string &posDataDir);

    // Load the negative examples returning the number of examples loaded (or negative on error)
    int loadNegatives( const string &negDataDir);

    // Train a classifier based on the RGB values of the extracts
    SVMClassifier::Ptr trainOnValue();

    // Train a classifier based on the range values of the extracts
    SVMClassifier::Ptr trainOnRange();


private:
    SVMParams svmp_;    // SVM training parameters

    list<ViewExtract::Ptr> pdata_;  // Training data (positives)
    list<ViewExtract::Ptr> ndata_;  // Training data (negatives)

    vector<cv::Mat> pexs_;    // Positive examples (e.g. colour or range images)
    vector<cv::Mat> nexs_;    // Negative examples (e.g. colour or range images)

    SVMClassifier::Ptr train();
    int loadData( const string&, list<ViewExtract::Ptr>&);
};  // end class

}   // end namespace

#endif
