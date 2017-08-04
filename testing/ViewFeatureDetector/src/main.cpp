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

#include <ViewFeatureDetector.h>    // RLearning
#include <FeatureDetector.h>    // RLearning
#include <CvModel.h>        // RLearning

#include <DataLoader.h> // RPascalVOC

#include <ViewFileReader.h> // RFeatures
#include <FeatureUtils.h>   // RFeatures
#include <ProHOG.h>     // RFeatures

#include <cstdlib>
#include <iostream>
using std::cerr;
using std::endl;
#include <fstream>
#include <algorithm>
#include <vector>
using std::vector;

#include <boost/foreach.hpp>


typedef vector<cv::Mat> TrainingSet;


// Load:
// Right facing car examples
// Front facing car examples
// Negative data
string loadData( const string &posDir, const string &aspects, const string &negDir,
               vector<cv::Mat> &rxs, vector<cv::Mat> &fxs, vector<cv::Mat> &negs)
{
    RPascalVOC::DataLoader pvocLoader( posDir, negDir, aspects);

    pvocLoader.loadPos( rxs, 2);    // Load right facing
    cerr << "Loaded right facing data: " << rxs.size() << endl;

    vector<cv::Mat> lxs;    // Load left facing
    pvocLoader.loadPos( lxs, 1);
    cerr << "Loaded left facing data" << endl;
    RFeatures::vertFlipReplace( lxs);   // Flip left facing examples to be right facing
    rxs.insert( rxs.end(), lxs.begin(), lxs.end()); // rxs now is all right facing
    cerr << "Flipped and collated data" << endl;

    //pvocLoader.loadPos( fxs, 3);    // Front facing
    //cerr << "Loaded front facing data: " << fxs.size() << endl;

    const int min = rxs.size();
    pvocLoader.loadNeg( negs, min);   // Negatives
    cerr << "Negatives loaded: " << negs.size() << endl;
    return "car";
}   // end loadData



cv::Mat_<int> extractProHOGFeatures( const vector<TrainingSet> &xs,
                    int nbins, const cv::Size &cdims, cv::Mat &trows)
{
    cerr << "Extracting feature vectors from " << xs.size() << " training sets..." << endl;
    cv::Mat_<int> labels(0,0,CV_32SC1);
    trows.create(0,0,CV_32FC1);

    int fvsz = 0;
    int dataCount = 0;
    int lab = -1;
    BOOST_FOREACH( const TrainingSet &ts, xs)
    {
        lab++;  // New dataset label
        BOOST_FOREACH( const cv::Mat &x, ts)
        {
            RFeatures::ProHOG prohog( x, nbins);
            const cv::Mat pfvRaw = prohog( cdims);
            cv::Mat pfv32;
            pfvRaw.convertTo( pfv32, CV_32F);
            const cv::Mat pfv = pfv32.reshape(1,1); // Single row CV_32FC1
            if ( fvsz == 0)
                fvsz = pfv.cols;
            assert( fvsz == pfv.cols);
            trows.push_back(pfv);
            labels.push_back(lab);
            dataCount++;
        }   // end foreach
    }   // end foreach

    assert( dataCount == trows.rows);
    cerr << "Extracted " << dataCount << " feature vectors of length " << fvsz << endl;
    assert( labels.rows == trows.rows);
    return labels;  // Single column vector
}   // end extractProHOGFeatures



int main( int argc, char *argv[])
{
    if ( argc != 5)
    {
        cerr << "Usage: " << argv[0] << " view_file pvoc_pos_dir pvoc_aspects pvoc_neg_dir" << endl;
        return EXIT_FAILURE;
    }   // end if

    const string posDir = argv[2];  // Positive examples
    const string aspects = argv[3]; // Example aspect notes
    const string negDir = argv[4];  // Negative images

    // Get the training sets
    vector<TrainingSet> xs;
    xs.push_back( vector<cv::Mat>());
    xs.push_back( vector<cv::Mat>());
    vector<cv::Mat> xstmp;  // Not used
    const string featureName = loadData( posDir, aspects, negDir, xs[1], xstmp, xs[0]);
    const cv::Size2f featureSize( 3.5, 2);
    const int nbins = 9;
    const cv::Size cdims( 5,3);
    cv::Mat trows;
    cv::Mat labels = extractProHOGFeatures( xs, nbins, cdims, trows);

    const View::Ptr view = RFeatures::ViewFileReader::load( argv[1]);

    // Define the feature extraction method (Pro-HOG) (calc pixel gradients over the image)
    RFeatures::ProHOG fop( view->img2d, nbins);
    fop.setCellDims( cdims);

    // Define and train the classifier
    //boost::shared_ptr<CvStatModel> cvMod( new CvKNearest( trows, labels));
    boost::shared_ptr<CvStatModel> cvMod( new CvSVM( trows, labels));
    RLearning::Model *model = new RLearning::CvModel( featureName, featureSize,
                                    fop.getFeatureType(), "CvSVM", cvMod);

    // Do the detection
    RLearning::FeatureDetector detector( model, &fop);
    RLearning::ViewFeatureDetector::Params params;
    params.step = 5;
    params.cutoff = 0;
    params.postCutoff = 0;
    params.edgeIntolerance = 1;
    params.threshProp = 1;
    params.power = 1;
    RLearning::ViewFeatureDetector vfd( view, &detector, params);
    vfd.detect();

    RFeatures::showImage( view->img2d, "Original image", false);

    // Display response map
    const cv::Mat_<float> rmap = vfd.getResponseMap();
    const cv::Mat drmap = RFeatures::convertForDisplay( rmap, true);
    RFeatures::showImage( drmap, "Response map", true);

    delete model;

    return EXIT_SUCCESS;
}   // end main
