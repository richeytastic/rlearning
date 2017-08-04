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

#include "ObjectDetectionStatsManager.h"
using RLearning::ObjectDetectionStatsManager;
using RLearning::ViewInfo;
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>

ObjectDetectionStatsManager::ObjectDetectionStatsManager()
{
    clearDetections();
    usePerViewNormalisedConfidenceScores( true);
}   // end ctor


void ObjectDetectionStatsManager::addComment( const std::string& comment) { _comments.push_back(comment);}
bool isSpace( char c) { return std::isspace(c);}


bool ObjectDetectionStatsManager::addView( int viewId, const cv::Size& vsz, const std::string& vstr)
{
    if ( hasView(viewId))  // View already exists?
        return false;

    // Replace any spaces in vstr with underscores
    std::string viewStr = vstr;
    std::replace_if( viewStr.begin(), viewStr.end(), isSpace, '_');
    _views[viewId] = ViewInfo( viewId, vsz, viewStr);
    _perViewMinConf[viewId] = FLT_MAX;
    _perViewMaxConf[viewId] = -FLT_MAX;
    return true;
}   // end addView


bool ObjectDetectionStatsManager::hasView( int vid) const { return _views.count(vid);}


int ObjectDetectionStatsManager::getNumViews() const { return _views.size();}


const ViewInfo* ObjectDetectionStatsManager::getView( int vid) const
{
    if ( !hasView( vid))
        return NULL;

    return &_views.at(vid);
}   // end getView


bool ObjectDetectionStatsManager::addGroundTruth( int viewId, const cv::Rect& gtbox)
{
    if ( !hasView(viewId))
    {
        std::cerr << "[ERROR] RLearning::ObjectDetectionStatsManager::addGroundTruth: Invalid view reference!" << std::endl;
        return false;
    }   // end if

    // Check against all other gtboxes for this view to make sure there's no overlap
    BOOST_FOREACH ( const cv::Rect& gt, _views[viewId].grndTrth)
    {
        if (( gt & gtbox).area() > 0)
        {
            std::cerr << "[ERROR] RLearning::ObjectDetectionStatsManager::addGroundTruth: Ground truth overlaps in same image!" << std::endl;
            return false;
        }   // end if
    }   // end foreach

    _views[viewId].grndTrth.push_back( gtbox);
    return true;
}   // end addGroundTruth



bool ObjectDetectionStatsManager::addGroundTruth( int vid, const std::vector<cv::Rect>& gtboxes)
{
    BOOST_FOREACH ( const cv::Rect& rct, gtboxes)
    {
        if ( !addGroundTruth(vid, rct))
            return false;
    }   // end foreach
    return true;
}   // end addGroundTruth



int ObjectDetectionStatsManager::loadGroundTruth( const std::string& gtfile, const cv::Size& vsz)
{
    std::vector<RFeatures::FeatureRecord> frecs;
    if ( !RFeatures::loadFeatureRecords( gtfile, frecs))
        return -1;

    int vids = 0;
    BOOST_FOREACH ( const RFeatures::FeatureRecord& frec, frecs)
    {
        std::istringstream iss( frec.viewInfo);
        int viewId;
        iss >> viewId;
        if ( addView( vids, vsz, RFeatures::PanoramaReader::createViewString( frec.dataId, viewId)))
            vids++;
    }   // end foreach

    return vids;
}   // end loadGroundTruth



int ObjectDetectionStatsManager::getNumGroundTruth( int* numViews) const
{
    int nv = 0;
    int numgt = 0;
    typedef std::pair<int, ViewInfo> VPair;
    BOOST_FOREACH ( const VPair& vp, _views)
    {
        const ViewInfo& view = _views.at(vp.first);
        numgt += view.grndTrth.size();
        if ( !view.grndTrth.empty())
            nv++;   // Yes, this view has some ground truth
    }   // end foreach

    if ( numViews)
        *numViews = nv;

    return numgt;
}   // end getNumGroundTruth


bool ObjectDetectionStatsManager::addDetection( int viewId, float conf, const cv::Rect& dbox)
{
    assert( !isnan(conf));

    if ( !hasView(viewId))
    {
        std::cerr << "[ERROR] RLearning::ObjectDetectionStatsManager::addDetection: Invalid view ID!" << std::endl;
        return false;
    }   // end if

    // Check against all other detection boxes for this view to make sure there's no overlap
    BOOST_FOREACH ( const cv::Rect& d, _views[viewId].detBoxes)
    {
        if (( d & dbox).area() > 0)
        {
            std::cerr << "[ERROR] RLearning::ObjectDetectionStatsManager::addDetection: Detections overlap in same image!" << std::endl;
            return false;
        }   // end if
    }   // end for

    _views[viewId].detBoxes.push_back(dbox);
    _views[viewId].detConfs.push_back(conf);

    // Set the per view min and max confidence
    if ( _perViewMinConf[viewId] > conf)
        _perViewMinConf[viewId] = conf;
    if ( _perViewMaxConf[viewId] < conf)
        _perViewMaxConf[viewId] = conf;

    // Set the all view min and max confidence
    if ( conf > _maxConf)
        _maxConf = conf;
    if ( conf < _minConf)
        _minConf = conf;
    return true;
}   // end addDetection


bool ObjectDetectionStatsManager::addDetections( int vid, const std::vector<float>& confScores, const std::vector<cv::Rect>& dboxes)
{
    const int nds = confScores.size();
    assert( nds == dboxes.size());
    for ( int i = 0; i < nds; ++i)
    {
        if ( !addDetection( vid, confScores[i], dboxes[i]))
            return false;
    }   // end for
    return true;
}   // end addDetections


void ObjectDetectionStatsManager::clearDetections()
{
    typedef std::pair<int, ViewInfo> VPair;
    BOOST_FOREACH ( const VPair& vp, _views)
    {
        ViewInfo& view = _views.at(vp.first);
        view.detBoxes.clear();
        view.detConfs.clear();
        _perViewMinConf[view.id] = FLT_MAX;
        _perViewMaxConf[view.id] = -FLT_MAX;
    }   // end foreach
    _minConf = FLT_MAX;
    _maxConf = -FLT_MAX;
    _usePerViewNormConf = false;
}   // end clearDetections


void ObjectDetectionStatsManager::usePerViewNormalisedConfidenceScores( bool enabled)
{
    _usePerViewNormConf = enabled;
}   // end usePerViewNormalisedConfidenceScores



void ObjectDetectionStatsManager::calcStats( double& tp, double& fn, double& tn, double& fp, double minConf) const
{
    assert( !isnan(minConf));
#ifndef NDEBUG
    if ( _usePerViewNormConf)
        assert( minConf >= 0 && minConf <= 1);
#endif
    tp = fn = tn = fp = 0;
    const int nviews = _views.size();

    typedef std::pair<int, ViewInfo> VPair;
    BOOST_FOREACH ( const VPair& vp, _views)
    {
        const ViewInfo& view = _views.at(vp.first);

        double gtarea = 0; // Find the total area of the ground truth boxes for this view
        BOOST_FOREACH ( const cv::Rect& gt, view.grndTrth)
            gtarea += gt.area();

        // Detection boxes don't overlap with each other, but each detection box might cover more than one gtbox
        // Only want those dets >= minCont
        double dtarea = 0; // Find the total area of the detection boxes in this view
        double intarea = 0; // Tot area of intersections of detections with ground truth boxes
        for ( int d = 0; d < view.detBoxes.size(); ++d)
        {
            double dconf = view.detConfs[d]; // Not per view normalised
            if ( _usePerViewNormConf)
                dconf = (dconf - _perViewMinConf.at(view.id)) / ( _perViewMaxConf.at(view.id) - _perViewMinConf.at(view.id));

            if ( dconf >= minConf)
            {
                const cv::Rect& dbox = view.detBoxes[d];
                dtarea += dbox.area();
                BOOST_FOREACH ( const cv::Rect& gt, view.grndTrth)
                    intarea += (gt & dbox).area();
            }   // end if
        }   // end for

        // Calculate for this view the tp, fn, tn, fp as ratios with the view area
        const double viewArea = view.win.width * view.win.height;
        const double vtp = intarea; // True positive pixels are covered by both ground truth and detection boxes
        const double vfn = gtarea - intarea;    // False neg pxls are ground truth boxes that don't intersect with detection boxes
        const double vfp = dtarea - intarea;    // False pos pxls are detection boxes that don't intersect with ground truth boxes
        const double vtn = viewArea - vtp - vfn - vfp;  // True negative pixels are everything else

        // Normalise these values by the image size when adding since images could be different sizes
        tp += vtp/viewArea;
        fn += vfn/viewArea;
        tn += vtn/viewArea;
        fp += vfp/viewArea;
    }   // end foreach

    const double pcntFactor = 100./nviews;
    // Divide through number of views and multiply by 100 for percentage scores
    tp *= pcntFactor;
    fn *= pcntFactor;
    tn *= pcntFactor;
    fp *= pcntFactor;
}   // end calcStats



double ObjectDetectionStatsManager::getMinThresh() const
{
    if ( _usePerViewNormConf)
        return 0;
    return _minConf;
}   // end getMinThresh



double ObjectDetectionStatsManager::getMaxThresh() const
{
    if ( _usePerViewNormConf)
        return 1;
    return _maxConf;
}   // end getMaxThresh



std::ostream& RLearning::operator<<( std::ostream& os, const ObjectDetectionStatsManager& odsm)
{
    // Add the comments
    BOOST_FOREACH ( const std::string& comment, odsm._comments)
        os << "#" << comment << std::endl;

    // Record each view 
    typedef std::pair<int, ViewInfo> ViewPair;
    BOOST_FOREACH ( const ViewPair& vp, odsm._views)
        os << vp.second << std::endl;
    return os;
}   // end operator<<


std::istream& RLearning::operator>>( std::istream& is, ObjectDetectionStatsManager& odsm)
{
    std::string ln;
    while ( std::getline( is, ln))
    {
        if ( ln.empty())
            continue;
        else if ( ln[0] == '#')
            odsm.addComment( ln.substr(1));
        else
        {
            std::istringstream viss(ln);
            ViewInfo view;
            viss >> view;
            odsm.addView( view.id, view.win, view.vstr);
            odsm.addGroundTruth( view.id, view.grndTrth);
            odsm.addDetections( view.id, view.detConfs, view.detBoxes);
        }   // end else
    }   // end while

    return is;
}   // end operator>>
