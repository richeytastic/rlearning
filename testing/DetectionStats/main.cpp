#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>
#include <iomanip>
using namespace std;

#include <PrecisionRecallFinder.h>
using RLearning::PrecisionRecallFinder;


void printData( const vector<float>& v1, const vector<float>& v2)
{
    assert( v1.size() == v2.size());
    const int sz = v1.size();
    for ( int i = 0; i < sz; ++i)
        cout << left << fixed << setw(5) << setprecision(4) << v1[i] << "  " << v2[i] << endl;
}   // end printData


void parseFile( ifstream& ifs, PrecisionRecallFinder& prf)
{
    float confidence, coveragePcnt;
    string ln;
    while ( getline( ifs, ln) && !ln.empty())
    {
        istringstream iss(ln);
        iss >> confidence >> coveragePcnt;
        prf.add( confidence, coveragePcnt);
    }   // end while
}   // end parseFile


void printUsageAndExit( char** argv)
{
    cerr << "Usage: " << argv[0] << " (0|1) maxConf confidence_coverage_file" << endl;
    cerr << "0 = ROC data (FPR vs Recall)" << endl;
    cerr << "1 = Precision/Recall data (1 - Precision vs Recall)" << endl;
    exit( EXIT_FAILURE);
}   // end printUsageAndExit



int main( int argc, char** argv)
{
    if ( argc != 4)
        printUsageAndExit( argv);

    const float maxConf = strtof( argv[2], 0);
    PrecisionRecallFinder prf( maxConf, 50);

    ifstream ifs;
    try
    {
        ifs.open(argv[3]);
        parseFile( ifs, prf);
    }   // end try
    catch ( const std::exception& e)
    {
        cerr << "Unable to parse file: " << argv[1] << endl;
        cerr << e.what() << endl;
        exit( EXIT_FAILURE);
    }   // end catch

    if ( ifs.is_open())
        ifs.close();

    vector<float> v1, v2;
    float v;
    const int ag = strtol( argv[1], 0, 10);
    switch ( ag)
    {
        case 0:
            v = prf.calcROCData( 100, v1, v2);
            cerr << "FPR TPR" << endl;
            printData( v1, v2);
            cerr << "Area Under Curve (AUC): " << v << endl;
            break;
        case 1:
            prf.calcPrecisionRecallData( 100, v1, v2);
            v = PrecisionRecallFinder::calcAveragePrecision( v1, v2);
            cerr << "Precision Recall" << endl;
            printData( v1, v2);
            cerr << "Average Precision (AP): " << v << endl;
            break;
        default:
            cerr << "Invalid argument!" << endl;
            printUsageAndExit( argv);
    }   // end switch

    return EXIT_SUCCESS;
}   // end main
