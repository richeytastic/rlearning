#pragma once
#ifndef RLEARNING_RANGE_PARTS_DETECTOR_H
#define RLEARNING_RANGE_PARTS_DETECTOR_H

#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>
#include <View.h>               // RFeatures
#include <FeatureUtils.h>       // RFeatures
#include <FeatureExtractor.h>   // RFeatures
#include <OffsetPatchScanner.h> // RFeatures
using RFeatures::FeatureExtractor;
using RFeatures::View;
#include "RealObjectSizeResponseSuppressor.h"   // RLearning
#include "Classification.h"     // RLearning
using RLearning::Classification;
using RLearning::Classifier;
#include <vector>


namespace RLearning
{

// Parts describe rectangularly bounded regions of objects. Since objects aren't rectangles,
// a part detector may include background material if at the edge of an object or representing
// the object as a whole.
struct PartDetector
{
    cv::Point2f propOffset;     // Offset to object reference point - given as proportion of part width/height from top left of part
    cv::Size2f realDims;        // Real size of part (in same units as range map)
    cv::Size minPxlDims;        // Minimum pixel resolution that a part shall be tested at (resized to fit min sampling dims of FX).
    double minValidRngCvg;      // If this part is entirely on the surface of the object, it should be 1
    float classifyThreshold;    // Classification responses >= this indicate possible presence of the part
    FeatureExtractor::Ptr fx;   // The feature extractor for the classifier (parts smaller than minPxlDims are scaled up to min sampling dims). 
    Classifier::Ptr classifier; // The corresponding classifier trained to detect this part
};  // end struct


class RangePartsDetector
{
public:
    RangePartsDetector( const View::Ptr v, float minRng, float maxRng, int stepSz=1);

    void registerPartDetector( const PartDetector& pd); // Also runs feature extractor pre-processing

    void generateResponses();

    int generateDetections( int maxDetections, float minResponseCutoff);

    const std::vector<cv::Rect>& getDetections() const { return _dboxes;}
    const std::vector<float>& getScores() const { return _dscores;}

    cv::Mat_<float> generateCombinedResponseMap() const;
    const cv::Mat_<float>& getPartResponseMap( int partId) const { return _responseMaps[partId];}

    const cv::Mat_<int>& getValidRangeCountIntImg() const { return _vrngCntii;}

    const cv::Mat_<byte>& getPlotMap() const { return _plotMap;}

private:
    const View::Ptr _view;
    const float _minRng;
    const float _maxRng;
    const int _stepSz;
    const int _responseResolution;

    std::vector<PartDetector> _partDetectors;   // First is always the whole object detector
    std::vector<RFeatures::Patch> _patches; // Corresponding patches for view scanning
    boost::unordered_map<std::string, FeatureExtractor::Ptr> _preProcessedFXs;
    std::vector< cv::Mat_<float> > _responseMaps;   // Corresponding response maps for the parts

    std::vector<cv::Rect> _dboxes;
    std::vector<float> _dscores;

    cv::Mat_<int> _vrngCntii;       // Integral image of pixels in valid range area

    cv::Mat_<byte> _plotMap;
};  // end class

}   // end namespace

#endif
