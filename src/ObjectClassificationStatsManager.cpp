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

#include "ObjectClassificationStatsManager.h"
using RLearning::ObjectClassificationStatsManager;
#include <cassert>


const cv::Size WINSZ(1,1);
const cv::Rect DBOX(0,0,WINSZ.width,WINSZ.height);


ObjectClassificationStatsManager::ObjectClassificationStatsManager() : ObjectDetectionStatsManager()
{
    usePerViewNormalisedConfidenceScores(false);    // Each classification has its own view
}   // end ctor


bool ObjectClassificationStatsManager::addPositive( int id)
{
    addView( id, WINSZ, "POS");
    return addGroundTruth( id, DBOX);
}   // end addPositive


bool ObjectClassificationStatsManager::addNegative( int id)
{
    return addView( id, WINSZ, "NEG");
}   // end addNegative


bool ObjectClassificationStatsManager::classifyPositive( int id, float confidence)
{
    assert( !isnan(confidence));
    return addDetection( id, confidence, DBOX);
}   // end classifyPositive


bool ObjectClassificationStatsManager::classifyNegative( int id, float confidence)
{
    assert( !isnan(confidence));
    return addDetection( id, confidence, DBOX);
}   // end classifyNegative


void ObjectClassificationStatsManager::clearClassifications()
{
    clearDetections();
}   // end clearClassifications


void ObjectClassificationStatsManager::calcStats( double& tp, double& fn, double& tn, double& fp, double minConf) const
{
    // Stats are returned from the detection stats manager as percentages.
    // We want them back as counts, so invert the percentage calc:
    ObjectDetectionStatsManager::calcStats( tp, fn, tn, fp, minConf);
    const double invFact = double(getNumViews())/100;
    tp *= invFact;
    fn *= invFact;
    tn *= invFact;
    fp *= invFact;
}   // end calcStats

