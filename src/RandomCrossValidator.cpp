#include "RandomCrossValidator.h"
using RLearning::RandomCrossValidator;
#include <cstdlib>


RandomCrossValidator::RandomCrossValidator( int tcnt, int ni,
                            const cv::Mat &xs, const cv::Mat &labs, int numEVs)
    : CrossValidator( xs, labs, numEVs), tcount_( tcnt < 1 ? 1 : tcnt), maxIts_( ni < 1 ? 1 : ni), iter_(0)
{
}   // end ctor


#include <iostream>
int RandomCrossValidator::createTrainingMask( const cv::Mat &labs, const vector<int> &counts, char *mask)
{
    static const int MAX_CLASHES = 3;

    for ( int i = 0; i < tcount_; ++i)
    {
        int randTry = MAX_CLASHES;
        // Negative set runs from index 0 to counts[0]
        int nidx = (int)(drand48() * counts[0]);
        while ( mask[nidx])
        {
            randTry--;
            if ( randTry > 0)
                nidx = (int)(drand48() * counts[0]);
            else
                nidx = (nidx + 1) % counts[0];
        }   // end while

        mask[nidx] = 1;

        randTry = MAX_CLASHES;
        // Positive set runs from counts[0] to labs.cols
        int pidx = (int)(drand48() * counts[1]) + counts[0];
        while ( mask[pidx])
        {
            randTry--;
            if ( randTry > 0)
                pidx = (int)(drand48() * counts[1]) + counts[0];
            else
                pidx = (pidx + 1) % counts[1] + counts[0];
        }   // end while

        mask[pidx] = 1;
    }   // end for
    
    iter_++;

    return 2*tcount_;
}   // end createTrainingMask



void RandomCrossValidator::printResults( ostream &os) const
{
    os << "[ Results after validation set " << iter_ << " of " << maxIts_ << " ]" << std::endl;
    const StatsGenerator* sgen = getStatsGenerator();
    double tp, fn, tn, fp;
    sgen->calcStats( tp, fn, tn, fp);
    Classification::printResultsTable( os, tp, fn, tn, fp);
}   // end printFinalResults


bool RandomCrossValidator::moreIterations() const
{
    return iter_ < maxIts_;
}   // end moreIterations
