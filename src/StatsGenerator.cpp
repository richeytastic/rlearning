#include "StatsGenerator.h"
using RLearning::StatsGenerator;
using RLearning::ClassificationMetricsGenerator;
using RLearning::ROCFinder;
using RLearning::Classification;
#include <cassert>
#include <cfloat>
#include <algorithm>
#include <iomanip>


void RLearning::writeColumnData( ostream& os, const std::vector<std::vector<double> >& cols)
{
    const int nrows = (int)cols[0].size();  // Lengths of all columns must match!
    const int ncols = (int)cols.size();
    for ( int i = 0; i < nrows; ++i)
    {
        for ( int j = 0; j < ncols; ++j)
        {
            const std::vector<double>& col = cols[j];
            assert( col.size() == nrows);
            os << std::fixed << std::setprecision(9) << std::setw(13) << col[i];
        }   // end for
        os << std::endl;
    }   // end for
}   // end writeColumnData


ROCFinder::ROCFinder()
    : _maxThresh(-FLT_MAX), _minThresh(FLT_MAX)
{}   // end ctor 


// private
void ROCFinder::updateThresholds( double val)
{
    _maxThresh = std::max<double>( _maxThresh, val);
    _minThresh = std::min<double>( _minThresh, val);
}   // end updateThresholds


void ROCFinder::classifiedPositive( double val)
{
    updateThresholds( val);
    _pvals.push_back( val);
}   // end classifiedPositive


void ROCFinder::classifiedNegative( double val)
{
    val = -val; // Negate
    updateThresholds( val);
    _nvals.push_back( val);
}   // end classifiedNegative


void ROCFinder::calcStats( double &tp, double &fn, double &tn, double &fp, double t) const
{
    tp = fn = tn = fp = 0;

    const int npvals = _pvals.size();
    for ( int i = 0; i < npvals; ++i)
    {
        if ( _pvals[i] >= t)
            tp += 1;
        else
            fn += 1;
    }   // end for

    const int nnvals = _nvals.size();
    for ( int i = 0; i < nnvals; ++i)
    {
        if ( _nvals[i] < t)
            tn += 1;
        else
            fp += 1;
    }   // end for
}   // end calcStats



/******************************************************************************************************/
ClassificationMetricsGenerator::ClassificationMetricsGenerator( const StatsGenerator* sgen) : _sgen(sgen) {}


double calcThresholdingData( int ndpts, const StatsGenerator* sgen, double mint, double maxt,
                             std::vector<double> &v0, Classification::Metric m0,
                             std::vector<double> &v1, Classification::Metric m1)
{
    assert( mint < maxt);
    double auc = 0;
    double a0 = 1, b0 = 1;

    v0.resize(ndpts);
    v1.resize(ndpts);
    const double stepSz = (maxt - mint)/(ndpts-1);
    for ( int i = 0; i < ndpts; ++i)
    {
        const double t = mint + i*stepSz; // Get the false positive and true negative ratios at threshold t
        double tp, fn, tn, fp;
        sgen->calcStats( tp, fn, tn, fp, t);
        v0[i] = Classification::calcMetric( tp, fn, tn, fp, m0);
        v1[i] = Classification::calcMetric( tp, fn, tn, fp, m1);

        // Sum the trapezoids for the area under the curve
        const double sumadd = (a0 - v0[i]) * (v1[i] + b0)/2;
        assert(sumadd >= 0);
        auc += sumadd;
        a0 = v0[i]; b0 = v1[i]; // For next iteration
    }   // end for

    return auc;
}   // end calcThresholdingData


double ClassificationMetricsGenerator::calcROCData( int ndpts, std::vector<double> &fprs, std::vector<double> &tprs) const
{
    const double mint = _sgen->getMinThresh();
    const double maxt = _sgen->getMaxThresh();
    return calcThresholdingData( ndpts, _sgen, mint, maxt, fprs, Classification::Fallout, tprs, Classification::Recall);
}   // end calcROCData


double ClassificationMetricsGenerator::calcPrecisionRecallData( int ndpts, std::vector<double> &precision, std::vector<double> &recall) const
{
    const double mint = _sgen->getMinThresh();
    const double maxt = _sgen->getMaxThresh();
    return calcThresholdingData( ndpts, _sgen, mint, maxt, precision, Classification::Precision, recall, Classification::Recall);
}   // end calcPrecisionRecallData



void ClassificationMetricsGenerator::calcThresholdVaryingMetric( int ndpts, Classification::Metric m, std::vector<double>& output) const
{
    const double mint = _sgen->getMinThresh();
    const double maxt = _sgen->getMaxThresh();
    assert( mint < maxt);

    output.resize(ndpts);
    const double stepSz = (maxt - mint)/(ndpts-1);
    for ( int i = 0; i < ndpts; ++i)
    {
        double tp, fn, tn, fp;
        _sgen->calcStats( tp, fn, tn, fp, mint + i*stepSz);
        output[i] = Classification::calcMetric( tp, fn, tn, fp, m);
    }   // end for
}   // end calcThresholdVaryingMetric


void ClassificationMetricsGenerator::calcThresholdVaryingMetrics( int ndpts, const std::vector<Classification::Metric>& ms,
                                                                  std::vector< std::vector<double> >& output) const
{
    output.resize(ms.size());
    for ( int i = 0; i < ms.size(); ++i)
        calcThresholdVaryingMetric( ndpts, ms[i], output[i]);
}   // end calcThresholdVaryingMetrics
