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
#ifndef RLEARNING_STATS_GENERATOR_H
#define RLEARNING_STATS_GENERATOR_H
#include "Classification.h"

namespace RLearning
{

void writeColumnData( ostream& os, const std::vector< std::vector<double> >& cols);


class ClassificationMetricsGenerator;   // Class definition below

class StatsGenerator
{
public:
    virtual void calcStats( double& tp, double& fn, double& tn, double& fp, double threshold=0) const = 0;

    virtual double getMinThresh() const = 0;
    virtual double getMaxThresh() const = 0;
};  // end class


class ClassificationMetricsGenerator
{
public:
    ClassificationMetricsGenerator( const StatsGenerator*);   // Can cast to this type

    // Collect ndpts many false positive ratios and true positive ratios datums.
    // Returns the area under the curve (AUC). Chart should be plotted with fprs as
    // the independent variable and tprs as the dependent variable.
    double calcROCData( int ndpts, std::vector<double> &fprs, std::vector<double> &tprs) const;

    // As above, but for precision and recall.
    double calcPrecisionRecallData( int ndpts, std::vector<double> &precision, std::vector<double> &recall) const;

    // Calculate other metrics over varying thresholds for ndpts datapoints.
    void calcThresholdVaryingMetric( int ndpts, Classification::Metric, std::vector<double>& output) const;

    void calcThresholdVaryingMetrics( int ndpts, const std::vector<Classification::Metric>&,
                                                    std::vector< std::vector<double> >& statcols) const;

private:
    const StatsGenerator* _sgen;
};  // end class


class ROCFinder : public StatsGenerator
{
public:
    ROCFinder();

    // Classify a positive or negative example with a given certainty
    // (not necessarily a probability). Values >= 0 indicate a classification that
    // is congruent with the function name and values < 0 indicate a false classification.
    // Values further from 0 (more negative or more positive) indicate greater classification certainty.
    void classifiedPositive( double val);
    void classifiedNegative( double val);
    virtual void calcStats( double& tp, double& fn, double& tn, double& fp, double threshold=0) const;

    virtual double getMinThresh() const { return _minThresh;}
    virtual double getMaxThresh() const { return _minThresh;}

private:
    double _maxThresh, _minThresh;
    std::vector<double> _pvals; 
    std::vector<double> _nvals;
    void updateThresholds(double);
};  // end class

}   // end namespace
#endif
