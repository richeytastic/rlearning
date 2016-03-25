#pragma once
#ifndef RLEARNING_REAL_OBJECT_SIZE_RESPONSE_SUPPRESSOR_H
#define RLEARNING_REAL_OBJECT_SIZE_RESPONSE_SUPPRESSOR_H

#include <AdaptiveDepthPatchScanner.h>  // RFeatures
#include <vector>

namespace RLearning
{

class RealObjectSizeResponseSuppressor
{
public:
    // Supply a range map and the real size of the object (in units commensurate with the range map).
    RealObjectSizeResponseSuppressor( const cv::Mat_<float>& rngMap, const cv::Size2f& realObjectDims);

    // Detects up to maxBoxes detections from the given responseMap. Note, that the response map is modified.
    // Returns the number of detections. Detection boxes and confidence scores can be retrieved using the helper
    // functions getDetectionBoxes and getDetectionResponses which will always return the boxes and corresponding
    // response values for the detections produced from the last call to this function.
    // By default, no responses <= 0 are detected. However, some classifiers (e.g., SVM)
    // may require minResponseCutoff to be set lower to include negative responses.
    // Also, the default is to assume that responses define the centre point of an object.
    // If this is not the case, set responseOffset to be a point (from 0 to 1 in both dimensions)
    // relative to the position of the response in the object being detected. For example, if
    // the middle base point of objects are being detected, responseOffset should be set to 0.5, 1.0.
    int getCandidateDetections( cv::Mat_<float> responseMap, int maxBoxes,
                                cv::Point2f responseOffset=cv::Point2f(0.5,0.5), float minResponseCutoff=0);

    const std::vector<cv::Rect>& getDetectionBoxes() const { return _dboxes;}
    const std::vector<float>& getDetectionResponses() const { return _dscores;}

    // The candidate detections may overlap. If no external validation is being done to discard multiples,
    // this function will suppress overlapping detections to include only the highest scoring detections.
    // Detections with lower scores that overlap have their bounding boxes adjusted so that they no longer
    // overlap with higher scoring detections. If however, a lower scoring detection has at least 50% of
    // its area overlapped by higher scoring detections, it is ignored completely.
    // Returns the number of detection rectangles/scores appended to the provided vectors.
    static int suppressOverlappingDetections( std::vector<cv::Rect>& inout, std::vector<float>& scores);

private:
    const cv::Mat_<float> _rngMap;
    const cv::Size2f _realObjectDims;
    std::vector<cv::Rect> _dboxes;
    std::vector<float> _dscores;
};  // end class

}   // end namespace
#endif
