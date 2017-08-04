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

#include "Classification.h"
using RLearning::Classification;
#include <iomanip>
using std::setw;
using std::fixed;
using std::setprecision;
#include <sstream>
using std::ostringstream;
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iomanip>



void Classification::printResultsTable( ostream &os, double tp, double fn, double tn, double fp)
{
    const double tot = tp + fn + fp + tn;    // Total count

    const double ppv = calcMetric( tp, fn, tn, fp, Precision);
    const double tpr = calcMetric( tp, fn, tn, fp, Recall);
    const double fpr = calcMetric( tp, fn, tn, fp, Fallout);    // Type I
    const double fnr = calcMetric( tp, fn, tn, fp, FNR);        // Type II
    const double acc = calcMetric( tp, fn, tn, fp, Accuracy);
    const double f1 = calcMetric( tp, fn, tn, fp, F1);
    const double mcc = calcMetric( tp, fn, tn, fp, MCC);

    const int PRC = 4;
    int CPRC = 0;   // Assume counts
    const double MIN_DIFF = 1e-9;
    if ( fabs(cvRound(tp) - tp) > MIN_DIFF || fabs(cvRound(fn) - fn) > MIN_DIFF || fabs(cvRound(tn) - tn) > MIN_DIFF || fabs(cvRound(fp) - fp) > MIN_DIFF)
        CPRC = 2;   // Percentages if not integers so use 2 decimal places

    // Set spaces for columns
    const int LGC = log10(tot) + 1;
    const int C1 = 19;
    const int C2 = 17;
    const int C3 = 26;
    const int C4 = std::max<int>(9,LGC);

    // Predicted positives row
    ostringstream tposs;
    tposs << "TP: " << setw(LGC) << fixed << setprecision(CPRC) << tp;
    ostringstream fposs;
    fposs << "FP: " << setw(LGC) << fixed << setprecision(CPRC) << fp;

    // Predicted negatives row
    ostringstream fnoss;
    fnoss << "FN: " << setw(LGC) << fixed << setprecision(CPRC) << fn;
    ostringstream tnoss;
    tnoss << "TN: " << setw(LGC) << fixed << setprecision(CPRC) << tn;

    // Right stats
    ostringstream ppvoss;
    ppvoss << "PPV (Precision) = " << fixed << setprecision(PRC) << ppv;
    ostringstream fnross;
    fnross << "  FNR (Type II) = " << fixed << setprecision(PRC) << fnr;
    ostringstream f1oss;
    f1oss << "              F1 = " << fixed << setprecision(PRC) << f1;

    // Bottom stats/totals
    ostringstream tpross;
    tpross << "TPR (Recall) = " << fixed << setprecision(PRC) << tpr;
    ostringstream fpross;
    fpross << "FPR (Type I) = " << fixed << setprecision(PRC) << fpr;
    ostringstream accoss;
    accoss << "Accuracy = " << fixed << setprecision(PRC) << acc << " (" << setprecision(CPRC) << (tp+tn) << "/" << tot << ")";
    ostringstream mccoss;
    mccoss << "MCC = " << fixed << setprecision(PRC) << mcc;

    os << fixed << setprecision(CPRC);
    using std::endl;
    os << std::right;
    os << setw(C1+C2)                              << "Real positives" << setw(C3) << "Real negatives" << setw(C4) <<  "Totals" << endl;
    os << setw(C1) << "Predicted positives" << setw(C2) << tposs.str() << setw(C3) <<      fposs.str() << setw(C4) << (tp + fp) << "   " << ppvoss.str() << endl;
    os << setw(C1) << "Predicted negatives" << setw(C2) << fnoss.str() << setw(C3) <<      tnoss.str() << setw(C4) << (tn + fn) << "   " << fnross.str() << endl;
    os << setw(C1) << "Totals"              << setw(C2) <<   (tp + fn) << setw(C3) <<        (tn + fp) << setw(C4) <<       tot << "  "  << f1oss.str() << endl;
    os << setw(C1+C2)                                  << tpross.str() << setw(C3) <<     fpross.str() << std::left << "                   " << accoss.str()
       << "\t" << mccoss.str() << endl;
}   // end printResultsTable



// static
double Classification::calcSSD( const cv::Mat &m1, const cv::Mat &m2)
{
    cv::Mat v1, v2;
    m1.convertTo( v1, CV_64F);
    m2.convertTo( v2, CV_64F);
    assert( v1.size() == v2.size() && v1.channels() == v2.channels());

    // Reshape to single channel, single row
    const cv::Mat v1rs = v1.reshape(1,1);
    const cv::Mat v2rs = v2.reshape(1,1);
    const double *v1Row = v1rs.ptr<double>(0);
    const double *v2Row = v2rs.ptr<double>(0);

    double sum = 0;
    const int cols = v1rs.cols;
    for ( int i = 0; i < cols; ++i)
        sum += pow(v1Row[i] - v2Row[i],2);
    return sum;
}   // end calcSSD



double calcPrecision( double tp, double fp)
{
    if ( tp == 0) return 0;
    return tp/(tp+fp);
}   // end calcPrecision

double calcNPR( double tn, double fn)
{
    if ( tn == 0) return 0;
    return tn/(tn + fn);
}   // end calcNPR

double calcRecall( double tp, double fn)  // TPR
{
    if ( tp == 0) return 0;
    return tp/(tp+fn);
}   // end calcRecall

double calcSpecificity( double tn, double fp) // TNR
{
    if ( tn == 0) return 0;
    return tn/(tn+fp);
}   // end calcSpecificity

double calcFallout( double fp, double tn) // FPR
{
    if ( fp == 0) return 0;
    return fp/(fp+tn);
}   // end calcFallout

double calcFNR( double fn, double tp)
{
    if ( fn == 0) return 0;
    return fn/(fn+tp);
}   // end calcFNR

double calcAccuracy( double tp, double tn, double fp, double fn)
{
    if ( tp + tn == 0) return 0;
    return (tp+tn)/(tp+tn+fp+fn);
}   // end calcAccuracy

double calcF1( double tp, double fp, double fn)  // F_1 score is harmonic mean of precision & recall 2.(p.r)/(p+r)
{
    const double p = calcPrecision( tp, fp);
    const double r = calcRecall( tp, fn);
    if ( p == 0 || r == 0) return 0;
    return (2.0*p*r)/(p+r);
}   // end calcF1

double calcMCC( double tp, double fn, double tn, double fp)
{
    double n = tn + tp + fn + fp;
    long double s = (tp + fn)/n;
    long double p = (tp + fp)/n;
    if ( s == 0 || p == 0 || s == 1 || p == 1) // Avoid div by 0
    {
        if ( tp == 0)
            return 0;
        else
            return 1;
    }   // end if
    return (tp/n - s*p)/sqrtl(p*s*(1.0 - s)*(1.0 - p));
}   // end calcMCC


// static
double Classification::calcMetric( double tp, double fn, double tn, double fp, Metric m)
{
    double v = 0;
    switch ( m)
    {
        case Precision:
            v = calcPrecision( tp, fp);
            break;
        case NPR:
            v = calcNPR( tn, fn);
            break;
        case Recall:
            v = calcRecall( tp, fn);
            break;
        case Specificity:
            v = calcSpecificity( tn, fp);
            break;
        case Fallout:
            v = calcFallout( fp, tn);
            break;
        case FNR:
            v = calcFNR( fn, tp);
            break;
        case Accuracy:
            v = calcAccuracy( tp, tn, fp, fn);
            break;
        case F1:
            v = calcF1( tp, fp, fn);
            break;
        case MCC:
            v = calcMCC( tp, fn, tn, fp);
            break;
        default:
            assert(false);
    }   // end switch
    return v;
}   // end calcMetric



void RLearning::writeData( ostream &os, const vector<cv::Mat_<float> > &points)
{
    if ( !os)
        return;

    using namespace std;
    BOOST_FOREACH( const cv::Mat_<float> &m, points)
    {
        cv::Mat_<float> mr = m.reshape(1,1);
        const int dims = mr.cols;
        if ( dims < 1)
            continue;

        const float *mrRow = mr.ptr<float>(0);
        const int dimsLess1 = dims - 1;
        os << fixed;
        for ( int i = 0; i < dimsLess1; ++i)
            os << mrRow[i] << " ";
        os << mrRow[dimsLess1] << std::endl;
    }   // end foreach
}   // end writeData



void RLearning::writeData( ostream &os, const cv::Mat_<float> &rowData)
{
    if ( !os)
        return;

    using namespace std;
    for ( int i = 0; i < rowData.rows; ++i)
    {
        cv::Mat_<float> mr = rowData.row(i).reshape(1,1);
        const int dims = mr.cols;
        if ( dims < 1)
            continue;

        const float *mrRow = mr.ptr<float>(0);
        const int dimsLess1 = dims - 1;
        os << fixed;
        for ( int i = 0; i < dimsLess1; ++i)
            os << mrRow[i] << " ";
        os << mrRow[dimsLess1] << std::endl;
    }   // end for
}   // end writeData



#include <string>
bool RLearning::readData( istream &is, vector<cv::Mat_<float> > &points)
{
    if ( !is)
        return false;

    bool gotError = false;

    std::string ln;
    int dims = -1;
    while ( std::getline( is, ln) && !ln.empty())
    {
        std::istringstream iss(ln);
        cv::Mat_<float> pt;
        while ( iss.good())
        {
            float a;
            iss >> a;
            pt.push_back(a);
        }   // end while

        if ( dims == -1)
            dims = pt.total();
        else
        {
            if ( dims != (int)pt.total())
            {
                std::cerr << "ERROR: Mismatch in point dimensions!" << std::endl;
                gotError = true;
                break;
            }   // end 
        }   // end else

        cv::Mat_<float> ptm = pt.reshape(1,1);
        points.push_back(ptm);
    }   // end while

    return gotError;
}   // end readData



bool RLearning::readData( istream &is, cv::Mat_<float> &rowData)
{
    if ( !is)
        return false;

    bool gotError = false;

    int i = 0;
    std::string ln;
    int dims = -1;
    while ( std::getline( is, ln) && !ln.empty())
    {
        std::istringstream iss(ln);
        cv::Mat_<float> pt;
        while ( iss.good())
        {
            float a;
            iss >> a;
            pt.push_back(a);
        }   // end while

        if ( dims == -1)
            dims = pt.total();
        else
        {
            if ( dims != (int)pt.total())
            {
                std::cerr << "ERROR: Mismatch in point dimensions!" << std::endl;
                gotError = true;
                break;
            }   // end 
        }   // end else

        rowData.row(i++) = pt.reshape(1,1);
    }   // end while

    return gotError;
}   // end readData
