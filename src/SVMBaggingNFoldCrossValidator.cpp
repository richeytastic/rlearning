#include "SVMBaggingNFoldCrossValidator.h"
using RLearning::SVMBaggingNFoldCrossValidator;
#include <cassert>
#include <cstdlib>


SVMBaggingNFoldCrossValidator::SVMBaggingNFoldCrossValidator( const SVMParams &svmp, int numc, int nf,
                    const cv::Mat_<float>& xs, const cv::Mat_<int>& labels, int numEVs)
    : NFoldCrossValidator( nf, xs, labels, numEVs),
    _kernel( svmp.makeKernel<cv::Mat_<float> >()), _cost(svmp.cost()), _eps(svmp.eps()),
    _nfolds(nf), _numClassifiers( numc < 2 ? 2 : numc)
{
}   // end ctor



// private - thread function
void SVMBaggingNFoldCrossValidator::trainGroup( int si, int numClassifiers,
                const vector<cv::Mat_<float> >* pset, const vector<cv::Mat_<float> >* nset)
{
    rlib::Random rnd0( si + numClassifiers);
    rlib::Random rnd1( 2*si + 2*numClassifiers + 2);
    const int fi = si + numClassifiers;
    for ( int i = si; i < fi; ++i)
    {
        vector<cv::Mat_<float> > tps, tns;   // This classifier's training data
        CrossValidator::sampleWithReplacement( *pset, tps, pset->size(), rnd0);
        CrossValidator::sampleWithReplacement( *nset, tns, nset->size(), rnd1);

#ifdef NDEBUG
        int nthreads = boost::thread::hardware_concurrency();
#else
        int nthreads = 1;
#endif
        SVMTrainer<cv::Mat_<float> > svmt( _kernel, _cost, _eps, nthreads);
        svmt.enableErrorOutput( false);
        _svmcs[i] = svmt.train( tps, tns);  // Train and set classifier
    }   // end for
}   // end trainGroup



void SVMBaggingNFoldCrossValidator::train( const cv::Mat_<float>& xs, const cv::Mat_<int>& labels)
{
    _svmcs.clear();  // Remove old classifiers
    _svmcs.resize(_numClassifiers);
    const int nc = _svmcs.size();

#ifdef NDEBUG
    int nthreads = boost::thread::hardware_concurrency();
#else
    int nthreads = 1;
#endif

    const int chunk = nc / nthreads;
    int rem = nc % nthreads;

    vector<cv::Mat_<float> > tpset, tnset;
    CrossValidator::splitIntoPositiveAndNegativeClasses( xs, labels, tpset, tnset);

    boost::thread_group tgroup;
    int si = 0;
    for ( int i = 0; i < nthreads; ++i)
    {
        // Size of thread chunk
        int tchunk = chunk;
        if ( rem > 0)
        {
            tchunk++;
            rem--;
        }   // end if

        tgroup.create_thread( boost::bind( &SVMBaggingNFoldCrossValidator::trainGroup, this, si, tchunk, &tpset, &tnset));
        si += tchunk;
    }   // end for

    tgroup.join_all();
}   // end train



float SVMBaggingNFoldCrossValidator::validate( const cv::Mat_<float> &x)
{
    double avgPredict = 0;
    BOOST_FOREACH( const SVMClassifier::Ptr svmc, _svmcs)
        avgPredict += svmc->predict(x);
    return float(avgPredict/_svmcs.size());
}   // end validate

