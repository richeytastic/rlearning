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

#pragma once
#ifndef RLEARNING_OBJECT_DETECTION_STATS_MANAGER_H
#define RLEARNING_OBJECT_DETECTION_STATS_MANAGER_H
#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>
#include <StatsGenerator.h> // RLearning
#include <DataTools.h>      // RFeatures
#include <ObjectDetectionStatsManager_ViewInfo.h>

namespace RLearning
{

class ObjectDetectionStatsManager : public StatsGenerator
{
public:
    ObjectDetectionStatsManager();

    void addComment( const std::string& comment);   // Comments are added in succession at top of file

    // Add a view. Returns false if view already exists.
    bool addView( int viewId, const cv::Size& winSz, const std::string& viewStr);
    bool hasView( int viewId) const;    // Is viewId already present?
    int getNumViews() const;
    const ViewInfo* getView( int viewId) const; // Return the view (non-mutable) or NULL if view doesn't exist.

    // Add a ground truth rectangle (specify the view ID to add it to). Returns false if viewId not present.
    // Overlapping ground truth are not allowed (will not add and will return false).
    bool addGroundTruth( int viewId, const cv::Rect& gtbox);
    bool addGroundTruth( int viewId, const std::vector<cv::Rect>& gtboxes);
    int getNumGroundTruth( int* vc=NULL) const;  // Return number of GT (and #views they're in using out param)

    // Load ground truth records from a file (using same format as for RFeatures::GroundTruthRecords).
    // Every view must have viewSize. Returns the number of views actually loaded or -1 on error.
    int loadGroundTruth( const std::string& gtfile, const cv::Size& viewSize);

    // Add a detection box (with confidence) to a view. Returns false if viewId not present.
    // Overlapping detections are not allowed (will not add and will return false).
    bool addDetection( int viewId, float confScore, const cv::Rect& dbox);
    bool addDetections( int viewId, const std::vector<float>& confScore, const std::vector<cv::Rect>& dboxes);

    void clearDetections(); // Reset to do a new round

    // Uses detection confidence scores that are normalised per view instead of the maximum and minimum
    // being defined over the entire set of detections from all views. True by default (since doing
    // multiple view object detection).
    void usePerViewNormalisedConfidenceScores( bool enabled=true);

    // Calculate the TP, FP, TN, FP as percentages over all detections having confidence scores >= minConfidence.
    virtual void calcStats( double& tp, double& fn, double& tn, double& fp, double minConfidence=0) const;

    virtual double getMinThresh() const;
    virtual double getMaxThresh() const;

private:
    std::vector<std::string> _comments;

    boost::unordered_map<int, ViewInfo> _views;         // Info for each view
    boost::unordered_map<int, float> _perViewMinConf;   // Per view minimum confidence levels
    boost::unordered_map<int, float> _perViewMaxConf;   // Per view maximum confidence levels
    float _minConf, _maxConf;       // Min and max confidence levels set in _dconfs over all views
    bool _usePerViewNormConf;       // Whether or not to use per view normalised detection confidence values

    friend std::ostream& operator<<( std::ostream&, const ObjectDetectionStatsManager&);
    friend std::istream& operator>>( std::istream&, ObjectDetectionStatsManager&);
};  // end class


// Stream operators
std::ostream& operator<<( std::ostream&, const ObjectDetectionStatsManager&);
std::istream& operator>>( std::istream&, ObjectDetectionStatsManager&);

}   // end namespace
#endif
