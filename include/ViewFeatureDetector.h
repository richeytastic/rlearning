/**
 * Common feature detection interface for 3-D images (encapsulated as View objects).
 * Defines image scanning method and defers detection itself to a FeatureDetector object.
 *
 * Richard Palmer
 * September 2012
 */

#pragma once
#ifndef RLEARNING_VIEW_FEATURE_DETECTOR
#define RLEARNING_VIEW_FEATURE_DETECTOR

//#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>

#include <View.h>
using RFeatures::View;
#include <DepthFinder.h>
using RFeatures::DepthFinder;
#include <AdaptiveDepthPatchScanner.h>  // RFeatures::PatchRanger
//#include <RegionSorter.h>
//using RFeatures::RegionSorter;
#include "FeatureDetector.h"
using RLearning::FeatureDetector;


namespace RLearning
{

class ViewFeatureDetector
{
public:
    // Optional parameter struct may be passed to the detection functions.
    struct Params;

    // Create a new ViewFeatureDetector with params:
    // v: The view to conduct detections over
    // model: The model to use for feature detections (must have previously been
    //        initialised with the same view data)
    // parms: Detection parameters (see struct declaration below)
    ViewFeatureDetector( const View::Ptr &v, FeatureDetector *model, const Params parms=Params());

    virtual ~ViewFeatureDetector(){}

    void reset();   // Resets the response and indicator maps
    void setParams( const Params&); // Set detection params (and reset)

    void detect();  // Run detector over the view

    // The raw detection response map (brighter -> stronger responses) (CV_32FC1)
    cv::Mat_<float> getResponseMap() const { return rspMap_;}
    // The indicator map of detection boxes (CV_8UC3)
    cv::Mat_<cv::Vec3b> getIndicatorMap() const { return indMap_;}
    // Return the number of times a detection box tested
    int getNumCalcs() const { return numTested_;}
    // Return the number of positive response locations
    int getPosCalcs() const { return posLocs_;}


    // Detection parameters
    struct Params
    {
        Params();

        int step;    // Pixel step size over image (larger -> coarser detections)

        // Raw response values less than this are treated as non instances
        // and are ignored. The true-positive ratio can be increased (at the
        // cost of increasing the false-positive ratio) by decreasing this
        // value. A value of zero is nominal.
        double cutoff;

        // Response values less than this AFTER POST PROCESSING are discarded
        // as non instances. This value is absolute so its setting depends on
        // the values of the other parameters.
        double postCutoff;

        // Object close to the image edge have their responses scaled by
        // the proportion of their area actually within the image. Increasing
        // this factor quadratically decreases the relative response of objects
        // at the edges. Setting this to zero removes this negative response
        // weighting entirely. Default value is 1. Relax this constraint by
        // using values between 0 and 1. (Responses are multiplied by
        // the ratio of the classifier model image's viewable area in the
        // image with the total area of the classifier model rectangle).
        // Negative values are allowed - but probably don't make much sense.
        double edgeIntolerance;

        // Proportion of positive response range to be considered as positive scores.
        // 1 means that 0 to the maximum repsonse is considered a positive score
        // (i.e. the default). 0.5 means that the range for a positive score goes
        // from half the max response up to the max response. Values greater
        // than 1 cause the positive score range to run from a negative value
        // (what is nominally a negative response) up to the maximum positive
        // response. Values less than 0 are meaningless.
        double threshProp;

        // Power to raise (un)exponentiated response to. If threshProp is used,
        // the raw value response is first translated to ensure no raw minimum
        // response values are used.
        double power;

        double minHeight, maxHeight; // Min and max height range
        double minDepth, maxDepth;   // Min and max depth range
    };  // end struct


private:
    FeatureDetector *model_;    // The feature classifier used for detections
    Params params_;             // Detection parameters
    View::Ptr view_;            // The view to search over
    DepthFinder::Ptr depthFinder_;  // Depth finder for the view
    cv::Rect imgRct_;           // Convenience rectangle defining view image bounds

    //boost::shared_ptr<RegionSorter> regionSorter_;    // Sort image responses
    cv::Mat_<float> rspMap_;        // CV_32FC1 response map same size as view
    cv::Mat_<cv::Vec3b> indMap_;    // Indicator map of response regions
    int numTested_;                 // Track number of tested positions in view
    int posLocs_;                   // Number of addResponse calls with positive responses


    bool setView( const View::Ptr&);    // View to run detections over

    // rct must be within image rectangle
    double testRect( const cv::Rect &rct) const;
    // Returns a value close to 1 if flat
    double testForFlatness( int row, int col) const;

    // Set a raw response on the response map over the provided area.
    // Before adding the response to the response map, the raw response value is
    // first modified according to the parameters as set. The process is as follows:
    // 1) If the raw response as provided is less than params.cutoff, the response
    // is assumed to be zero (and so nothing is added to the response map) and
    // zero is returned.
    // 2) Responses >= params.cutoff are translated back to zero.
    // 3) The new response is raised by the exponent params.power
    // and this final value is added to the response map.
    void addResponse( double rawResponse, const cv::Rect &area);
};  // end class

}   // end namespace

#endif
