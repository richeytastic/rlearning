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

#include <ProHOGTools.h>        // RFeatures
#include <PascalVOCHOGUtils.h>  // RPascalVOC
#include <PCA.h>                // RLearning

#include <iostream>
using std::cout;
using std::endl;
#include <fstream>
#include <iomanip>

#define TESTING 1



void readFeatVecParams( RPascalVOC::FeatParams &fvp)
{
    std::ifstream ifs( "feat_params.cfg");
    ifs >> fvp;
    cout << fvp;
    ifs.close();
}   // end readFeatVecParams



void printDataSizes( const vector<cv::Mat> &pos, const vector<cv::Mat> &neg, int ncnt)
{
    cout << "Total number of raw positive examples = " << pos.size() << endl;
    cout << "Total number of raw negative examples = " << neg.size() << endl;
    if ( neg.size() < ncnt)
        cout << "WARNING: Requested number of negative examples ("
             << ncnt << ") could not be produced!" << endl;
}   // end printDataSizes



void printEigenvectors( const cv::Mat &eigenVals, const cv::Mat &eigenRows, int top)
{
    // Eigen vectors as columns
    for ( int i = 0; i < top; ++i)
    {
        cout << "Eigenvector (eigenvalue = " <<  eigenVals.at<double>(i,0) << ")" << endl;
        //RLearning::printMatrix( eigenRows.row(i), cout);
    }   // end for
}   // end printEigenvectors



// Eigenvectors as rows
void projectToEigenvectors( const RLearning::PCA &pcaPos, const cv::Mat &evecsRowsPos,
                            const RLearning::PCA &pcaNeg, const cv::Mat &evecsRowsNeg)
{
    cv::Mat selEvecsRowsPos( 2, evecsRowsPos.cols, evecsRowsPos.type());
    cv::Mat selEvecsRowsNeg( 2, evecsRowsNeg.cols, evecsRowsNeg.type());
    evecsRowsPos.row(0).copyTo( selEvecsRowsPos.row(0));
    evecsRowsNeg.row(0).copyTo( selEvecsRowsNeg.row(0));

    for ( int i = 0; i < 10; ++i)
    {
        evecsRowsPos.row(i+1).copyTo( selEvecsRowsPos.row(1));
        evecsRowsNeg.row(i+1).copyTo( selEvecsRowsNeg.row(1));

        std::ostringstream oss;
        oss << i+1;
        const string rowVals = oss.str();

        cv::Mat posProjColVecs = pcaPos.project( selEvecsRowsPos);
        const string posfname = string("pos_pts_") + rowVals + string(".txt");
        std::ofstream ofs1( posfname.c_str());
        RLearning::writePoints( ofs1, (cv::Mat_<double>)posProjColVecs, true);
        ofs1.close();

        cv::Mat negProjColVecs = pcaNeg.project( selEvecsRowsNeg);
        const string negfname = string("neg_pts_") + rowVals + string(".txt");
        std::ofstream ofs2( negfname.c_str());
        RLearning::writePoints( ofs2, (cv::Mat_<double>)negProjColVecs, true);
        ofs2.close();
    }   // end for
}   // end projectToEigenvectors



int main( int argc, char* argv[])
{
    srand48( TESTING ? 1 : time(NULL));

    PascalVOCParser::Ptr parser = RPascalVOC::PascalVOCParser::create( argc, argv);
    if ( parser == NULL)
        return EXIT_FAILURE;

    cout << "EIGENVECTOR FINDER";
    if ( TESTING)
        cout << " [TESTING]";
    cout << endl;

    const string cls( argv[2]);

    RPascalVOC::FeatParams fvp;
    readFeatVecParams( fvp);

    cout << "Retrieving image sets..." << endl;
    vector<cv::Mat> posImgs;
    vector<cv::Mat> negImgs;
    const int negCount = 1382;//15000; // 1382
    parser->getClassImagery( cls, negCount, posImgs, negImgs);
    printDataSizes( posImgs, negImgs, negCount);

    cout << "Extracting feature vectors..." << endl;
    const cv::Size cellDims( fvp.cellsWide, fvp.cellsHigh);

    RFeatures::BatchProHOGExtractor posExtractor( posImgs, fvp.numBins, fvp.dirDep, cellDims, fvp.useDiff);
    vector<cv::Mat> phogs;
    posExtractor.extract_mt( phogs);
    cout << "Total positive feature vectors = " << phogs.size() << endl;
    RFeatures::BatchProHOGExtractor negExtractor( negImgs, fvp.numBins, fvp.dirDep, cellDims, fvp.useDiff);
    vector<cv::Mat> nhogs;
    negExtractor.extract_mt( nhogs);
    cout << "Total negative feature vectors = " << nhogs.size() << endl;

    cout << "Calculating eigenvectors for positive feature vectors..." << endl;
    RLearning::PCA posPCA( phogs);
    cv::Mat evecsRowsPos;
    cv::Mat evalsColPos = posPCA.calcEigenvectors( evecsRowsPos);   // Takes time

    cout << "Calculating eigenvectors for negative feature vectors..." << endl;
    RLearning::PCA negPCA( nhogs);
    cv::Mat evecsRowsNeg;
    cv::Mat evalsColNeg = negPCA.calcEigenvectors( evecsRowsNeg);   // Takes time

    projectToEigenvectors( posPCA, evecsRowsPos, negPCA, evecsRowsNeg);

    return EXIT_SUCCESS;
}   // end main
