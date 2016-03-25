#include "RealObjectSizeResponseSuppressor.h"
using RLearning::RealObjectSizeResponseSuppressor;
#include <cassert>


RealObjectSizeResponseSuppressor::RealObjectSizeResponseSuppressor( const cv::Mat_<float>& rngMap, const cv::Size2f& realObjectDims)
    : _rngMap(rngMap), _realObjectDims(realObjectDims)
{}   // end ctor


int RealObjectSizeResponseSuppressor::getCandidateDetections( cv::Mat_<float> responseMap, int numBoxes, cv::Point2f respOffset, float minResponseCutoff)
{
    cv::Mat_<float> respMap = responseMap;
    assert( respMap.size() == _rngMap.size());
    _dboxes.clear();
    _dscores.clear();
    const cv::Size realObjDims = _realObjectDims;

    cv::Rect imgRect( 0, 0, respMap.cols, respMap.rows);
    cv::Point mnp, mxp;
    double mn, mx;

    const RFeatures::PatchRanger pranger( _rngMap);

    const float ZERO_VAL = minResponseCutoff;
    while ( numBoxes > 0)
    {
        cv::minMaxLoc( respMap, &mn, &mx, &mnp, &mxp);
        if ( mx <= ZERO_VAL)
            break;

        // Get the size of the object at the detection point
        cv::Rect scaledRect;
        pranger.calcPatchRect( mxp, realObjDims, scaledRect);   // scaledRect centred over mxp
        respMap.at<float>(mxp) = ZERO_VAL; // Zero out this response so we don't retrieve it again
        // Shift the rectangle by the relative position defined by the response offset
        scaledRect.x += (0.5 - respOffset.x) * scaledRect.width;
        scaledRect.y += (0.5 - respOffset.y) * scaledRect.height;
        cv::Rect dbox = scaledRect & imgRect;   // Ensure stays in image

        if ( dbox.area() > 0)
        { // Responses are added in order of most confident first
            _dboxes.push_back(dbox);
            _dscores.push_back(mx);
            numBoxes--;
        }   // end if
    }   // end while

    return _dboxes.size();
}   // end getCandidateDetections


int RealObjectSizeResponseSuppressor::suppressOverlappingDetections( std::vector<cv::Rect>& nboxes, std::vector<float>& nscores)
{
    assert( nboxes.size() == nscores.size());

    std::vector<cv::Rect> outboxes;
    std::vector<float> outscores;

    const int ntmp = (int)nboxes.size();

    // Detection boxes may overlap. Look at each box in turn (from least confident to most) and if it intersects
    // a detection having higher confidence, we reduce the dimensions of this box so there's no overlap. The
    // existing (higher scoring) detection remains unchanged.
    for ( int i = ntmp-1; i >= 0; --i)
    {
        bool addBox = true;
        cv::Rect dbox = nboxes[i];
        const int origBoxArea = dbox.area();
        assert( origBoxArea > 0);

        const int halfOrigBoxArea = origBoxArea/2;

        for ( int j = i-1; j >= 0; --j) // Test for overlaps against the other, higher scoring detections
        {
            const cv::Rect& tmpBox = nboxes[j];
            const cv::Rect intRect = dbox & tmpBox;    // Intersecting rectangle

            const int intArea = intRect.area();
            if ( intArea == 0)
                continue;

            // Detections are deemed to overlap if a detection with a lower score overlaps higher
            // score detection boxes with at least 50% of its area.
            if ( (dbox.area() - intArea) < halfOrigBoxArea)
            {
                addBox = false;
                break;
            }   // end if

            // Adjust width or height?
            if (( double(intRect.height) / dbox.height) > ( double(intRect.width) / dbox.width))
            {
                // dbox overlaps on the right or left?
                if ( dbox.x > tmpBox.x)   // Right
                {
                    const int newx = tmpBox.x + tmpBox.width;
                    dbox.width -= newx - dbox.x;
                    dbox.x = newx;
                }   // end if
                else    // Left
                    dbox.width = tmpBox.x - dbox.x;
            }   // end if
            else
            {
                // dbox overlaps on the bottom or top?
                if ( dbox.y > tmpBox.y) // Bottom
                {
                    const int newy = tmpBox.y + tmpBox.height;
                    dbox.height -= newy - dbox.y;
                    dbox.y = newy;
                }   // end if
                else    // Top
                    dbox.height = tmpBox.y - dbox.y;
            }   // end else
        }   // end for

        if ( addBox && dbox.area() > 0)
        {
            outboxes.push_back(dbox);
            outscores.push_back(nscores[i]);
        }   // end if
    }   // end for

    nboxes = outboxes;
    nscores = outscores;
    return nboxes.size();
}   // end suppressOverlappingDetections
