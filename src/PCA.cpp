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

#include "PCA.h"
#include <cassert>
using std::vector;
using std::ostream;


cv::Mat RLearning::flattenToColumnVectors( const vector<cv::Mat> &data)
{
    assert( !data.empty());
    const int ndims = data[0].rows * data[0].cols * data[0].channels();
    cv::Mat rowVecs( data.size(), ndims, CV_32FC1);

    int i = 0;
    BOOST_FOREACH( const cv::Mat &datum, data)
    {
        assert( datum.isContinuous());
        cv::Mat d = datum.reshape(1,1); // Single channel, single row
        d.copyTo( rowVecs.row(i++));
    }   // end foreach

    return rowVecs.t();
}   // end flattenToColumnVectors



void RLearning::separateRowVectors( const cv::Mat &rowVecs, vector<cv::Mat> &vs)
{
    for ( int i = 0; i < rowVecs.rows; ++i)
    {
        const cv::Mat m = rowVecs.row(i);
        vs.push_back(m);
    }   // end for
}   // end separateRowVectors



cv::Mat RLearning::calcMedian( const vector<cv::Mat> &c)
{
    assert( !c.empty());
    const int fvLen = c[0].rows * c[0].cols * c[0].channels();
    const int numData = c.size();
    cv::Mat dataMat( numData, fvLen, CV_32FC1);
    for ( int i = 0; i < numData; ++i)
    {
        const cv::Mat rowVec = c[i].reshape(1,1);
        assert( rowVec.type() == CV_32FC1);
        rowVec.copyTo( dataMat.row(i));
    }   // end foreach

    // dataMat now has rows of training data (dimensions in columns)
    // Now need to sort each column independently.
    cv::Mat sortedMat( dataMat.size(), dataMat.type());
    cv::sort( dataMat, sortedMat, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

    const int midx = numData/2;   // Gives middle row if numData is odd
    const float *row0 = sortedMat.ptr<float>(midx);
    float *row1 = 0;
    if ( numData % 2 == 0)
        row1 = sortedMat.ptr<float>(midx-1);

    cv::Mat med(1, fvLen, CV_32FC1);
    float *outRow = med.ptr<float>(0);

    for ( int i = 0; i < fvLen; ++i)
    {
        outRow[i] = row0[i];
        if ( numData % 2 == 0)
            outRow[i] = (row0[i] + row1[i])/2;
    }   // end for

    return med;
}   // end calcMedian



cv::Mat RLearning::calcMeans( const vector<cv::Mat> &data)
{
    assert( !data.empty());
    const int channels = data[0].channels();
    cv::Mat means = cv::Mat::zeros( data[0].size(), CV_32FC(channels));

    BOOST_FOREACH( const cv::Mat &m, data)
    {
        assert( m.channels() == channels);
        assert( m.size() == means.size());
        means += m;   // Sum
    }   // end foreach

    means /= data.size(); // Average
    return means;
}   // end calcMeans



cv::Mat RLearning::calcMeans( const cv::Mat &colVecs)
{
    assert( !colVecs.empty());
    cv::Mat rowVecs = colVecs.t();
    cv::Mat means = cv::Mat::zeros( 1, rowVecs.cols, CV_32FC(colVecs.channels()));

    int sz = rowVecs.rows;
    for ( int i = 0; i < sz; ++i)
        means += rowVecs.row(i);

    means /= sz; // Average
    return means.t();
}   // end calcMeans



cv::Mat RLearning::calcCovariance( const cv::Mat &colVecs, bool sampleBias, cv::Mat means)
{
    const int numData = colVecs.cols;
    assert( numData > 1);
    const cv::Mat trowVecs = colVecs.t();
    const cv::Mat rowVecs = trowVecs.reshape(1, numData);

    // Ensure means vector is calculated
    if ( means.empty())
        means = calcMeans( colVecs);

    cv::Mat meanVec = means.reshape(1,1); // Single channel, single row
    const int type = meanVec.type();
    assert( type == CV_32FC1);
    assert( rowVecs.type() == type);

    const int sqDim = meanVec.cols;
    cv::Mat_<float> covMat = cv::Mat_<float>::zeros( sqDim, sqDim);

    const float *meanRow = meanVec.ptr<float>(0);

    for ( int t = 0; t < numData; ++t)
    {
        const float *dRow = rowVecs.ptr<float>(t);

        for ( int i = 0; i < sqDim; ++i)
        {
            const float *mi = &meanRow[i];
            const float *di = &dRow[i];

            float *cvRow1 = covMat.ptr<float>(i);

            for ( int j = i; j < sqDim; ++j)
            {
                const float *mj = &meanRow[j];
                const float *dj = &dRow[j];

                float *cvDat1 = &cvRow1[j]; // Output in first position in matrix
                float *cvRow2 = covMat.ptr<float>(j);
                float *cvDat2 = &cvRow2[i]; // Output in second (symmetric) position

                const float v = (*di - *mi)*(*dj - *mj);
                *cvDat1 += v;
                if ( i != j)    // Symmetric position
                    *cvDat2 += v;
            }   // end for
        }   // end for
    }   // end for

    covMat /= sampleBias ? numData - 1 : numData;
    return covMat;
}   // end calcCovariance



cv::Mat RLearning::calcCovariance( const vector<cv::Mat> &data, bool sampleBias, cv::Mat means)
{
    assert( !data.empty());

    // Ensure means vector is calculated
    if ( means.empty())
        means = calcMeans( data);

    cv::Mat meanVec = means.reshape(1,1); // Single channel, single row
    const int type = meanVec.type();
    assert( type == CV_32FC1);
    cv::Mat covMat = cv::Mat::zeros( cv::Size( meanVec.cols, meanVec.cols), type);
    
    const int sqDim = meanVec.cols;

    const float*meanRow = meanVec.ptr<float>(0);

    BOOST_FOREACH( const cv::Mat &d, data)
    {
        const cv::Mat d1 = d.reshape(1,1);  // Single channel, single row
        assert( d.type() == type);
        assert( d1.size() == meanVec.size());

        const float *dRow = d1.ptr<float>(0);

        for ( int i = 0; i < sqDim; ++i)
        {
            const float *mi = &meanRow[i];
            const float *di = &dRow[i];

            float *cvRow1 = covMat.ptr<float>(i);

            for ( int j = i; j < sqDim; ++j)
            {
                const float *mj = &meanRow[j];
                const float *dj = &dRow[j];

                float *cvDat1 = &cvRow1[j]; // Output in first position in matrix
                float *cvRow2 = covMat.ptr<float>(j);
                float *cvDat2 = &cvRow2[i]; // Output in second (symmetric) position

                const float v = (*di - *mi)*(*dj - *mj);
                *cvDat1 += v;
                if ( i != j)    // Symmetric position
                    *cvDat2 += v;
            }   // end for
        }   // end for
    }   // end for

    const int dataSz = data.size();
    covMat /= sampleBias ? dataSz - 1 : dataSz;
    return covMat;
}   // end calcCovariance



void RLearning::printMatrix( const cv::Mat &m, ostream &os)
{
    const int channels = m.channels();
    assert( m.depth() == CV_32F);

    for ( int i = 0; i < m.rows; ++i)
    {
        const float *mRow = m.ptr<float>(i);

        for ( int j = 0; j < m.cols; ++j)
        {
            const float *mDat = &mRow[j*channels];

            os << "[";
            for ( int k = 0; k < channels; ++k)
            {
                os << mDat[k];
                if ( k < channels - 1)
                    os << ",";
            }   // end for - channels

            os << "]";
            if ( j < m.cols - 1)
                os << "; ";
        }   // end for - columns

        os << std::endl;
    }   // end for - rows
}   // end printMatrix



void RLearning::printColumnVectors( const cv::Mat &data, ostream &os)
{
    for ( int i = 0; i < data.cols; ++i)
    {
        os << i << ") ";
        printMatrix( data.col(i).t(), os);  // Print as a row vector
    }   // end for
}   // end printColumnVectors



void RLearning::writePoints( ostream &os, const cv::Mat_<float> &data, bool colOrder)
{
    cv::Mat rowVecs = data;
    if ( colOrder)
        rowVecs = data.t();

    const int rows = rowVecs.rows;
    const int cols = rowVecs.cols;
    for ( int i = 0; i < rows; ++i)
    {
        const float *rowi = rowVecs.ptr<float>(i);
        for ( int j = 0; j < cols; ++j)
        {
            os << rowi[j];
            if ( j < cols - 1) os << " ";
        }   // end for
        os << std::endl;
    }   // end for
}   // end writePoints



using RLearning::PCA;

PCA::PCA( const vector<cv::Mat> &data, bool useSampleBias)
{
    useSampleBias_ = useSampleBias;
    assert( !data.empty());
    origChannels_ = data[0].channels();
    origRows_ = data[0].size().height;
    colVecs_ = flattenToColumnVectors( data);
    calcMeans();
}   // end ctor



PCA::PCA( const cv::Mat &colData, bool useSampleBias)
{
    useSampleBias_ = useSampleBias;
    cv::Mat rowVecs = colData.t();  // Data as rows

    origChannels_ = rowVecs.channels();
    origRows_ = rowVecs.cols;
    const int numData = rowVecs.rows;

    assert( numData > 0);

    colVecs_.create( origRows_ * origChannels_, numData, CV_32FC1);

    for ( int i = 0; i < numData; ++i)
    {
        cv::Mat vi = rowVecs.row(i);
        assert( vi.isContinuous());
        cv::Mat cvec = vi.reshape(1,0).t(); // Column vector
        assert( cvec.rows == origRows_ * origChannels_);
        cvec.copyTo( colVecs_.col(i));
    }   // end for

    calcMeans();
}   // end ctor



void PCA::calcMeans()
{
    means_ = RLearning::calcMeans( colVecs_);
    // Subtract means from the data
    const int sz = colVecs_.cols;
    for ( int i = 0; i < sz; ++i)
        colVecs_.col(i) -= means_;
}   // end calcMeans



cv::Mat PCA::getMeans()
{
    return means_;
}   // end getMeans



cv::Mat PCA::getCovariance()
{
    if ( covMat_.empty())
        covMat_ = RLearning::calcCovariance( colVecs_, useSampleBias_, means_);
    return covMat_;
}   // end getCovariance



cv::Mat PCA::calcEigenvectors( cv::Mat &evRows)
{
    if ( eigenVals_.empty())
    {
        cv::Mat covMat = getCovariance();
        eigenVals_.create( covMat.rows, 1, covMat.type());
        eigenRows_.create( covMat.size(), covMat.type());
        cv::eigen( covMat, eigenVals_, eigenRows_);
    }   // end if

    evRows = eigenRows_;
    return eigenVals_;
}   // end calcEigenvectors



void PCA::project( const cv::Mat &evRows, cv::Mat &outColVecs) const
{
    outColVecs = evRows * colVecs_;   // Original data in columns
}   // end project


cv::Mat PCA::project( const cv::Mat &evRows) const
{
    return evRows * colVecs_;   // Original data in columns
}   // end project



void PCA::reconstruct( const cv::Mat &pColData, const cv::Mat &evRows, cv::Mat &outColVecs) const
{
    const cv::Mat rDataRows = pColData.t() * evRows;
    outColVecs = rDataRows.t();
    // Add means back on
    const int sz = outColVecs.cols;
    for ( int i = 0; i < sz; ++i)
        outColVecs.col(i) += means_;
}   // end reconstruct



cv::Mat PCA::reconstruct( const cv::Mat &pColData, const cv::Mat &evRows) const
{
    const cv::Mat rDataRows = pColData.t() * evRows;
    cv::Mat retCols = rDataRows.t();
    // Add means back on
    const int sz = retCols.cols;
    for ( int i = 0; i < sz; ++i)
        retCols.col(i) += means_;
    return retCols;
}   // end reconstruct
