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

#include "CrossValidator.h"
using RLearning::CrossValidator;
#include <cassert>
#include <iostream>


CrossValidator::CrossValidator( const cv::Mat_<float> &xs, const cv::Mat_<int> &labs, int numEVs)
    : _numEVs( numEVs), _txs(xs), _tlabels(labs)
{
    assert( _tlabels.total() == _txs.rows);

    // Ensure labs are given as a single row vector
    if ( labs.rows > labs.cols)
        _tlabels = labs.t();

    // Count the number of entries of each class
    const int* labArray = _tlabels.ptr<int>(0);
    for ( int i = 0; i < _tlabels.cols; ++i)
    {
        const int lab = labArray[i];
        while ( lab >= (int)_cCounts.size())
            _cCounts.push_back(0);
        _cCounts[lab]++;
    }   // end for

    // At the moment, CrossValidator can only deal with two class problems
    if ( _cCounts.size() != 2)
        std::cerr << "ERROR: Currently CrossValidator can only deal with 2 class problems!" << std::endl;
    assert( _cCounts.size() == 2);

    if ( _numEVs <= 0)
        _numEVs = 0;
    if ( _numEVs > _txs.cols)
        _numEVs = _txs.cols;
}   // end ctor



void CrossValidator::processAll()
{
    while ( next());
}   // end processAll



bool CrossValidator::next()
{
    if ( !moreIterations())
        return false;

    // Get the training instance indices for this cross-val iteration
    char *tidxs = (char*)calloc( _tlabels.cols, sizeof(char));  // Training (1) / validation (0) indices
    const int numt = createTrainingMask( _tlabels, _cCounts, tidxs);

    // Create subset of data to train on
    const int tidxCnt = _tlabels.cols;
    cv::Mat_<float> trows( numt, _txs.cols); // Training rows
    cv::Mat_<int> tlabs( 1, numt);    // Training labels
    int j = 0;
    vector<int> vids;  // Store the IDs for doing validation
    for ( int i = 0; i < tidxCnt; ++i)
    {
        if ( tidxs[i])
        {
            _txs.row(i).copyTo( trows.row(j));
            tlabs.at<int>(0,j) = _tlabels.at<int>(0,i);
            j++;
        }   // end if
        else
            vids.push_back(i);
    }   // end for

    // Do PCA on the training data if needed and project using the number of requested eigenvectors
    if ( _numEVs > 0)
    {
        RLearning::PCA pca( trows.t(), true);   // Data as column vectors
        cv::Mat evecs;  // Eigenvectors as rows
        cv::Mat evals = pca.calcEigenvectors( evecs);
        const cv::Mat basisRows = evecs.rowRange( 0, _numEVs);
        const cv::Mat trowst = trows.t();
        trows = (basisRows * trowst).t();    // Training data projected and transposed to rows
        //const cv::Mat txst = _txs.t();
        //txs = (basisRows * txst).t();  // All data projected and transposed to rows
    }   // end if

    train( trows, tlabs);

    // Classify over the validation set
    BOOST_FOREACH( const int& i, vids)
    {
        const float v = this->validate( _txs.row(i));
        if ( _tlabels.at<int>(0,i) == 0)
            _rocFinder.classifiedNegative( -v);
        else
            _rocFinder.classifiedPositive( v);
    }   // end for

    free( tidxs);
    return moreIterations();
}   // end next



void CrossValidator::getClassCounts( vector<int> &ccnts) const
{
    ccnts = _cCounts;
}   // end getClassCounts



// static
void CrossValidator::splitIntoPositiveAndNegativeClasses( const cv::Mat_<float>& xs, const cv::Mat_<int>& labels,
                                                          vector<cv::Mat_<float> >& pset,
                                                          vector<cv::Mat_<float> >& nset)
{
    const int *labsVec = labels.ptr<int>(0);
    for ( int i = 0; i < xs.rows; ++i)
    {
        assert( labsVec[i] == 0 || labsVec[i] == 1);
        if (labsVec[i] == 1)
            pset.push_back(xs.row(i));
        else if (labsVec[i] == 0)
            nset.push_back(xs.row(i));
    }   // end for
}   // end splitIntoPositiveAndNegativeClasses



// static
cv::Mat_<float> CrossValidator::createCrossValidationMatrix( const vector< const vector<cv::Mat_<float> >* >& rowVectors,
                                                             cv::Mat_<int>& labels)
{
    assert( !rowVectors.empty());
    cv::Mat_<float> vecs;
    labels.create( 0,1);
    for ( int label = 0; label < rowVectors.size(); ++label)
    {
        assert( rowVectors[label] != 0);
        const vector<cv::Mat_<float> >& rvecs = *rowVectors[label];
        assert( !rvecs.empty());
        const int colDim = rvecs[0].cols; // Should be the length of each row vector in this class
        if ( vecs.empty())
            vecs.create(0, colDim);
        assert( colDim == vecs.cols);   // Ensure this class's row vector length matches what's already stored

        for ( int i = 0; i < rvecs.size(); ++i)
        {
            const cv::Mat_<float>& rv = rvecs[i];
            if ( rv.rows != 1 || rv.cols != colDim)
            {
                std::cerr << "ERROR feature vector size: " << rv.size() << std::endl;
                assert( rv.rows == 1 && rv.cols == colDim);
            }   // end if

            vecs.push_back( rv);    // Append the row vector to the bottom of the matrix
            labels.push_back(label);    // Set this vector's class label
        }   // end for
    }   // end for

    labels = labels.t();    // Make row vector
    return vecs;
}   // end createCrossValidationMatrix



// static
cv::Mat_<float> CrossValidator::createCrossValidationMatrix( const vector<cv::Mat_<float> >& negRowVectors,
                                                             const vector<cv::Mat_<float> >& posRowVectors,
                                                             cv::Mat_<int>& labels)
{
    vector< const vector<cv::Mat_<float> >* > rowVectors(2);
    rowVectors[0] = &negRowVectors;
    rowVectors[1] = &posRowVectors;
    return createCrossValidationMatrix( rowVectors, labels);
}   // end createCrossValidationMatrix



// static
void CrossValidator::sampleWithReplacement( const vector<cv::Mat_<float> > &pop,
                                                  vector<cv::Mat_<float> > &sample, int sz, rlib::Random& rnd)
{
    const int psz = pop.size();
    for ( int i = 0; i < sz; ++i)
    {
        const int idx = rnd.getRandomInt() % psz;
        sample.push_back( pop[idx]);
    }   // end for
}   // end sampleWithReplacement



// static
unordered_set<int> CrossValidator::sampleWithoutReplacement( const vector<cv::Mat_<float> > &pop,
                                                             vector<cv::Mat_<float> > &sample, int sz, rlib::Random& rnd)
{
    const int psz = pop.size();
    assert( sz <= psz);
    unordered_set<int> idxs;

    for ( int i = 0; i < sz; ++i)
    {
        int idx = rnd.getRandomInt() % psz;
        int clashCount = psz / sz; // Allowed several clashes (already indexed elements)
        while ( idxs.count(idx))
        {
            if ( clashCount > 0)
            {
                idx = rnd.getRandomInt() % psz;  // Another chance for a random entry
                clashCount--;
            }   // end if
            else
                idx = (idx+1) % psz;
        }   // end while
        sample.push_back( pop[idx]);
        idxs.insert(idx);
    }   // end for

    return idxs;
}   // end sampleWithoutReplacement



