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

#include "RangePartsDetector.h"
using RLearning::RangePartsDetector;
using RLearning::PartDetector;
#include <algorithm>
#include <cmath>
#include <cassert>


RangePartsDetector::RangePartsDetector( const View::Ptr v, float minRng, float maxRng, int stepSz)
    : _view(v), _minRng(minRng), _maxRng(maxRng), _stepSz(stepSz), _responseResolution(4)
{
    // To allow measurement of proportion of valid range coverage
    cv::Mat_<byte> scanMask;
    _vrngCntii = RFeatures::createMaskIntegralImage<float>( _view->rngImg, minRng, maxRng, scanMask);
}   // end ctor


void RangePartsDetector::registerPartDetector( const PartDetector& inpd)
{
    // Set the view scanning patch for this part
    RFeatures::Patch patch;
    patch.minPxlDims = inpd.minPxlDims;
    patch.realDims = inpd.realDims;
    patch.propOffset = inpd.propOffset;
    _patches.push_back(patch);

    // Create a part detector with a pre-processed feature extractor for the view.
    // The view is only processed once for parts using the same feature extractor.
    PartDetector pd = inpd;
    const std::string fxstring = pd.fx->getConstructString();
    if ( !_preProcessedFXs.count(fxstring))
        _preProcessedFXs[fxstring] = pd.fx->preProcess(_view);  // Get the correct image type for the feature extractor
    pd.fx = _preProcessedFXs[fxstring];
    _partDetectors.push_back(pd);
}   // end registerPartDetector



// public
void RangePartsDetector::generateResponses()
{
    const cv::Size& imgsz = _view->size();
    const int respRes = _responseResolution;

    _plotMap = cv::Mat_<byte>::zeros( imgsz.height/respRes, imgsz.width/respRes);

    // Scan the range map to generate candidate rectangles for the parts
    RFeatures::OffsetPatchScanner patchScanner( _view->rngImg, _patches, _stepSz);
    patchScanner.scan( _minRng, _maxRng);
    const int nparts = _partDetectors.size();
    _responseMaps.clear();

    for ( int parti = 0; parti < nparts; ++parti)
    {
        const PartDetector& pd = _partDetectors[parti];
        cv::Mat_<float> respMap = cv::Mat_<float>::zeros( imgsz.height/respRes, imgsz.width/respRes);

        // Get the candidate detection patches for this part - all are guaranteed to be wholly contained in the image.
        const std::list<RFeatures::OffsetPatch>& opatches = patchScanner.getOffsetPatches(parti);

        // Any patch rectangles smaller than the min sampling dims, use the raw image extract resized and reprocessed.
        // This may be less efficient than working on the FX pre-processed view (depending on the feature extractor being
        // used), but it is better than missing out entirely on a potential detection just because the feature extractor
        // requires a given pixel resolution.
        const cv::Size& minSamplingDims = pd.fx->getMinSamplingDims();

        // Do classification for each feature vector extracted from an offset patch,
        // setting the response at the object reference part accordingly.
        BOOST_FOREACH ( const RFeatures::OffsetPatch& opatch, opatches)
        {
            // If this patch rectangle doesn't also cover enough of the valid range, ignore.
            const int vrngCnt = RFeatures::getIntegralImageSum<int>( _vrngCntii, opatch.pxlRect);
            const double rngCvg = double(vrngCnt) / opatch.pxlRect.area();
            if ( rngCvg < pd.minValidRngCvg)
                continue;

            cv::Mat_<float> fv;
            if ( opatch.pxlRect.width < minSamplingDims.width || opatch.pxlRect.height < minSamplingDims.height)
                fv = pd.fx->extract( opatch.pxlRect, minSamplingDims); // opatch dims too small for the feature extractor so resize
            else
                fv = pd.fx->extract( opatch.pxlRect);

            assert( !fv.empty());

            const float v = pd.classifier->predict( fv);
            if ( v < pd.classifyThreshold)
                continue;

            cv::Point opt = opatch.pxlPt;    // Object reference point for normal size view
            // Scale down the position of the reference point for the response map dimensions
            // Change the scale of the offset and the rectangle centre to fit into the response maps
            opt.x = int(double(opt.x)/respRes);
            opt.y = int(double(opt.y)/respRes);
            assert( RFeatures::isWithin( respMap, opt));
            respMap.at<float>( opt) = v - pd.classifyThreshold; // Always positive
            _plotMap.at<byte>( opt) = 255;
        }   // end for - all candidate parts classified

        // Smooth and normalise the response map for this part to create a probability distribution
        cv::GaussianBlur( respMap, respMap, cv::Size(7,7), 0, 0);
        cv::Mat_<float> normedRespMap( respMap.size());
        cv::normalize( respMap, normedRespMap, 1, 0, cv::NORM_L1);
        _responseMaps.push_back( normedRespMap);
    }   // end for
}   // end generateResponses



// public
cv::Mat_<float> RangePartsDetector::generateCombinedResponseMap() const
{
    // Combine the individual response maps into a single one
    const cv::Size& imgsz = _view->size();
    cv::Mat_<float> respMap = cv::Mat_<float>::zeros( imgsz.height/_responseResolution, imgsz.width/_responseResolution);
    const int nparts = _responseMaps.size();
    for ( int i = 0; i < nparts; ++i)
        respMap += _responseMaps[i];
    return respMap;
}   // end generateCombinedResponseMap



// public
int RangePartsDetector::generateDetections( int maxDetections, float minResponseCutoff)
{
    if ( _responseMaps.empty())
        return 0;

    cv::Mat_<float> respMap = generateCombinedResponseMap();

    // Find candidate objects from the whole object response map
    cv::Mat_<float> scaledRng; // Scale down the view range image
    cv::resize( _view->rngImg, scaledRng, respMap.size());

    const PartDetector& rootPart = _partDetectors[0];
    RLearning::RealObjectSizeResponseSuppressor suppressor( scaledRng, rootPart.realDims);
    suppressor.getCandidateDetections( respMap, maxDetections, rootPart.propOffset, minResponseCutoff);
    std::vector<cv::Rect> boxes = suppressor.getDetectionBoxes();
    std::vector<float> scores = suppressor.getDetectionResponses();
    RLearning::RealObjectSizeResponseSuppressor::suppressOverlappingDetections( boxes, scores);

    const int nb = boxes.size();
    for ( int i = 0; i < nb; ++i)   // Scale up the boxes
        RFeatures::scale( boxes[i], _responseResolution);

    _dboxes = boxes;
    _dscores = scores;

    /*
    // Validate
    _dboxes.clear();
    _dscores.clear();
    for ( int i = 0; i < nb; ++i)
    {
        const cv::Rect& dbox = boxes[i];
        _dboxes.push_back( dbox);
        _dscores.push_back( scores[i]);
    }   // end for
    */

    return _dboxes.size();
}   // end generateDetections
