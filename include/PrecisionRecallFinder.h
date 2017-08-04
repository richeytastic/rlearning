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

/**
 * Use to determine statistics for object detection (bounding box location).
 * Each detection is represented by the confidence in the detection and a percentage
 * coverage of the bounding box. Coverage of a bounding box > TP_COV_PCNT (default 50%)
 * gives a true positive.
 *
 * Richard Palmer
 * July 2013
 */

#pragma once
#ifndef RLEARNING_PRECISION_RECALL_FINDER_H
#define RLEARNING_PRECISION_RECALL_FINDER_H

#include "Classification.h"
#include <vector>
using std::vector;


namespace RLearning
{

class PrecisionRecallFinder
{
public:
    PrecisionRecallFinder( float maxConf, float true_positive_coverage_pcnt=50);

    // Clears the scores while keeping the maxConf
    void clear();

    // confidence must be non-negative.
    // covPcnt must be in range [0,100]
    void add( float confidence, float covPcnt);

    // Gives steps+1 data points for precision and recall.
    // Plot precision as independent variable.
    // May subsequently pass output vectors to calcAveragePrecision.
    void calcPrecisionRecallData( int steps, vector<float>& precision, vector<float>& recall) const;

    // See Pascal VOC Challenge for details.
    static float calcAveragePrecision( const vector<float>& precision, const vector<float>& recall);

    // Gives n data points for false-positives and true-positives.
    // Returns the Area Under the Curve (AUC).
    // Plot FPR as independent variable.
    float calcROCData( int n, vector<float>& fprs, vector<float>& tprs) const;

private:
    float _maxConf;
    const float TP_COV_PCNT; // True positives are > this (default is 50% coverage)

    struct Score
    {
        Score( float c, float p) : confidence(c), coveragePcnt(p) {}

        float confidence;
        float coveragePcnt;

        bool operator<( const Score& s) const
        {
            return this->confidence < s.confidence;
        }   // end operator<
    };  // end struct

    mutable vector<Score> _scores;

    void calcRatios( float confThresh, int&, int&, int&, int&) const;
};  // end class

}   // end namespace

#endif
