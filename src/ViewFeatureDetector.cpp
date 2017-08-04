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

#include "ViewFeatureDetector.h"
using RLearning::ViewFeatureDetector;
#include <cmath>
#include <iostream>
using std::cerr;
using std::endl;
#include <cfloat>
#include <cassert>


ViewFeatureDetector::Params::Params()
    : step(1), cutoff(0), edgeIntolerance(0), threshProp(0), power(1),
    minHeight(FLT_MIN), maxHeight(FLT_MAX), minDepth(0), maxDepth(FLT_MAX)
{}   // end ctor



ViewFeatureDetector::ViewFeatureDetector( const View::Ptr &v, FeatureDetector *m, const Params p)
    : model_(m), params_(p)
{
    assert( model_ != NULL);
    reset();
    assert( v != NULL);
    setView( v);
}   // end ctor



void ViewFeatureDetector::reset()
{
    numTested_ = 0;
    posLocs_ = 0;
    rspMap_.release();
    indMap_.release();
    depthFinder_.reset();
}   // end reset



void ViewFeatureDetector::setParams( const Params &p)
{
    reset();
    params_ = p;
}   // end setParams



// private
bool ViewFeatureDetector::setView( const View::Ptr &v)
{
    reset();
    view_ = v;
    imgRct_ = cv::Rect(0,0,0,0);

    if ( view_ != NULL)
    {
        const cv::Size sz = view_->img2d.size();
        assert( sz.width == sz.height);
        //regionSorter_ = boost::shared_ptr<RegionSorter>(new RegionSorter( sz.width));
        imgRct_ = cv::Rect( 0, 0, view_->img2d.cols, view_->img2d.rows);
        depthFinder_ = DepthFinder::Ptr( new DepthFinder( view_->rngImg));
    }   // end if

    return view_ != NULL;
}   // end setView



void ViewFeatureDetector::detect()
{
    assert( view_ != NULL);
    int step = params_.step;
    if ( step < 1) step = 1;

    const cv::Size vsz = view_->img2d.size();
    rspMap_ = cv::Mat_<float>::zeros( vsz);   // Reset response map
    indMap_ = cv::Mat_<cv::Vec3b>::zeros( vsz); // Reset indicator map

    const int minImgDim = std::min( imgRct_.width, imgRct_.height);
    if ( step > minImgDim)
        step = minImgDim;

    const cv::Rect viewRect = imgRct_;

    RFeatures::PatchRanger patchRanger( view_->rngImg);
    const cv::Size2f actSz = model_->getActualSize();   // Dimensions of object in metres

    cv::Rect estSz; // Estimated size at a given location in the view
    for ( int i = 0; i < viewRect.height; i += step)
    {
        // Scan pattern alternates as left to right in even rows and right to left in odd rows
        if ( i % 2 == 0)
        {   // Even row
            for ( int j = 0; j < viewRect.width; j += step)   // scan left to right
            {
                // Calculate the rectangle with centre at i,j having the correct scale
                patchRanger.calcPatchRect( i, j, actSz, estSz);
                const cv::Rect intRect = estSz & viewRect;
                double rsp = testRect( intRect);
                addResponse( rsp, intRect);  // Post process response
            }   // end for
        }   // end if
        else 
        {   // Odd row
            for ( int j = viewRect.width - 1; j >= 0; j -= step)   // scan right to left
            {
                // Calculate the rectangle with centre at i,j having the correct scale
                patchRanger.calcPatchRect( i, j, actSz, estSz);
                const cv::Rect intRect = estSz & viewRect;
                double rsp = testRect( intRect);
                addResponse( rsp, intRect);  // Post process response
            }   // end for
        }   // end else
    }   // end for - rows
}   // end detect



double ViewFeatureDetector::testRect( const cv::Rect &rct) const
{
    /*
    // Continue if estimated model size is too big or small (in proportion to the image)
    const double mhProp = (double)rct.height/imgRct_.height;
    const double mwProp = (double)rct.width/imgRct_.width;
    if ( mhProp < MIN_MODEL_PROP || mhProp > MAX_MODEL_PROP ||
         mwProp < MIN_MODEL_PROP || mwProp > MAX_MODEL_PROP)
        continue;
    */

    const double minDepth = 3;  // 3 metres minimum
    // Get the average depth over the model rectangle. Adjusts dimensions of provided
    // rectangle to be that portion that intersects the image and that the depth value
    // was averaged over.
    if ( depthFinder_->getAvgDepth( rct) < minDepth)
        return 0; // Ignore areas of the image too close in

    // We also ignore patches that overlap areas where no range data is available
    // (since these areas might contain sky or areas that might otherwise give too
    // high a response value for the range change).
    if ( view_->rngImg.at<float>( rct.y + rct.height/2, rct.x + rct.width/2) == 0 // middle of patch
        || view_->rngImg.at<float>( rct.y, rct.x) == 0  // top left
        || view_->rngImg.at<float>( rct.y, rct.x + rct.width) == 0   // top right
        || view_->rngImg.at<float>( rct.y + rct.height, rct.x) == 0     // bottom left
        || view_->rngImg.at<float>( rct.y + rct.height, rct.x + rct.width) == 0) // bottom right
        return 0;

    /*
    // What is the range change in depth over this extract? We don't care
    // about the orientation of the change so we use the final channel of the view's
    // range gradient integral image. Take the square root because changes in range
    // increase with distance.
    double rangeChange = (*view->rngGrads)( imRect, view->rngGrads->channels()-1)/(imRect.width*imRect.height);
    // We ignore regions with too high range changes since these areas are too noisy to be reliable
    if ( rangeChange > 10)
        return;
    */

    // Don't detect unless intersection of the rectangle and the image
    // has dimensions at least as large as the classifier image.
    double rsp = model_->detect( rct);  // Do the detection. Values >= are a positive detection.


    /*
    // Larger than normal range changes will be correlated with the boundary of an object
    // as will larger than normal changes between detections (remember this algorithm scans in a
    // continuous line). We multiply these values together since they are correlated:
    double correlatedEdge = 0.7*rsp * 0.3*rangeChange;
    */

    return rsp;
}   // end testRect



double ViewFeatureDetector::testForFlatness( int row, int col) const
{
    const double d0z = depthFinder_->getDepth( row, col);
    if ( d0z == 0)
        return 0;

    /*
    const double halfRows = imgRct_.height / 2;
    const double r0 = row - halfRows;
    const double f = (double)halfRows;    // Only works for 90 degree FOV
    const double d0h = d0z * r0/f;

    static const double delta = 2;
    // Calculate (by similar triangles) the image row indices for 2 vertically adjacent points
    const int r1 = (int)(f * d0h/(d0z - delta) + halfRows + 0.5);   // Assumed further in
    const int r2 = (int)(f * d0h/(d0z + delta) + halfRows + 0.5);   // Assumed further away
    */

    const int r1 = row - 1;
    const int r2 = row + 1;
    if ( r1 < 0 || r1 >= imgRct_.height || r2 < 0 || r2 >= imgRct_.height)
        return 0;

    const int col1 = col - 1;
    const int col2 = col + 1;
    if ( col1 < 0 || col1 >= imgRct_.width || col2 < 0 || col2 >= imgRct_.width)
        return 0;

    // Get the two new depth values
    const double d1z = depthFinder_->getDepth( r1, col);
    const double d2z = depthFinder_->getDepth( r2, col);
    const double d3z = depthFinder_->getDepth( row, col1);
    const double d4z = depthFinder_->getDepth( row, col2);
    if ( d1z == 0 || d2z == 0 || d3z == 0 || d4z == 0)  // No depth value so can't do calc
        return 0;

    const double c1 = d1z - d2z;  // Depth change vertically
    const double c2 = d3z - d4z;  // Depth change horizontally

    const double flatness = pow(c1 + c2,2);
    cerr << flatness << endl;
    return flatness;
}   // end testForFlatness



void ViewFeatureDetector::addResponse( double rawResp, const cv::Rect &area)
{
    numTested_++;
    if ( rawResp < params_.cutoff)
        return;

    rawResp -= params_.cutoff;
    rawResp = pow( rawResp, params_.power);

    rspMap_( area) += rawResp;   // Add to the response map
    cv::rectangle( indMap_, area, cv::Scalar(200,50,0)); // Draw on indicator map

    /*
    double* minMax = regionSorter_->add( area, rawResp); // Add to the region sorter
    maxResp_ = minMax[1];
    minResp_ = minMax[0];
    */

    posLocs_++;
}   // end addResponse


