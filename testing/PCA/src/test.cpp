#include "PCA.h"
#include <cstdlib>
#include <iostream>
#include <string>
using std::string;
#include <fstream>
#include <sstream>
#include <vector>
using std::vector;



// Returns true on successful read
bool readPoints( const string &fname, vector< vector<double> > &points)
{
    bool gotError = false;

    std::ifstream ifs( fname.c_str());
    string ln;
    int dims = -1;
    while ( std::getline( ifs, ln) && !ln.empty())
    {
        std::istringstream iss(ln);
        vector<double> pt;
        while ( iss.good())
        {
            double a;
            iss >> a;
            pt.push_back(a);
        }   // end while

        if ( dims == -1)
            dims = pt.size();
        else
        {
            if ( dims != pt.size())
            {
                std::cerr << "ERROR: Mismatch in point dimensions!" << std::endl;
                gotError = true;
                break;
            }   // end 
        }   // end else

        points.push_back(pt);
    }   // end while

    ifs.close();
    return !gotError;
}   // end readPoints



cv::Mat collateData( const string &fname)
{
    vector< vector<double> > points;
    if ( !readPoints( fname, points))
    {
        std::cerr << "Invalid point data - exiting..." << std::endl;
        exit(1);
    }   // end if

    if ( points.empty())
    {
        std::cerr << "No data read! - exiting..." << std::endl;
        exit(1);
    }   // end if

    int numPoints = points.size();
    int numDims = points[0].size();
    cv::Mat data( numPoints, numDims, CV_64FC1);

    for ( int i = 0; i < numPoints; ++i)
    {
        const vector<double> &v = points[i];
        double *dataRow = data.ptr<double>(i);
        for ( int j = 0; j < numDims; ++j)
            dataRow[j] = v[j];
    }   // end for

    return data.t();
}   // end collateData



int main( int argc, char **argv)
{
    if ( argc != 2)
    {
        std::cerr << "Please provide filename of point data!" << std::endl;
        exit(1);
    }   // end if

    const string dataFile( argv[1]);
    cv::Mat colVecs = collateData( dataFile);    // Data as column vectors

    std::cout << "Original data (as row vectors)" << std::endl;
    RLearning::printColumnVectors( colVecs, std::cout);

    RLearning::PCA pca( colVecs);

    cv::Mat means = pca.getMeans();
    std::cout << "\nDimension means" << std::endl;
    RLearning::printMatrix( means.t(), std::cout);

    cv::Mat covMat = pca.getCovariance();
    std::cout << "\nCovariance matrix" << std::endl;
    RLearning::printMatrix( covMat, std::cout);

    // Find eigenvalues and eigenvectors
    cv::Mat eigenRows;   // Row vectors
    cv::Mat eigenVals = pca.calcEigenvectors( eigenRows);   // Values as column vector

    // Eigen vectors as columns
    for ( int i = 0; i < eigenVals.rows; ++i)
    {
        std::cout << "\nEigenvector (eigenvalue = " <<  eigenVals.at<double>(i,0) << ")" << std::endl;
        RLearning::printMatrix( eigenRows.row(i), std::cout);
    }   // end for

    // Derive the new data in terms of these eigenvectors

    // Projected data is in columns
    const cv::Mat tDataCols = pca.project( eigenRows);
    std::cout << "\nProjected data (as row vectors)" << std::endl;
    RLearning::printColumnVectors( tDataCols, std::cout);

    // Back project to the original data
    const cv::Mat rData = pca.reconstruct( tDataCols, eigenRows);

    std::cout << "\nReconstructed data (as row vectors)" << std::endl;
    RLearning::printColumnVectors( rData, std::cout);

    return EXIT_SUCCESS;
}   // end main
