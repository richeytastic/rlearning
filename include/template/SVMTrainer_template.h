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

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <sstream>
using std::ostringstream;
#include <iomanip>
#include <sys/time.h>


template <typename T>
const double SVMTrainer<T>::TAU = 1e-12;


template <typename T>
struct SVMTrainer<T>::Alpha
{
    uint idx;
    double alpha;

    Alpha( uint i, SVMTrainer<T> *s) : idx(0), alpha(0), svm(s)
    {
        update(i);
    }   // end ctor

    void update( uint i)
    {
        svm->alphas[idx] = alpha;
        idx = i;
        alpha = svm->alphas[i];
    }   // end reset

private:
    SVMTrainer<T> *svm;
};  // end struct Alpha


template <typename T>
class SVMTrainer<T>::ThreadFn
{
public:
    ThreadFn( double aH, double aL, uint I, uint J, uint K, uint sgSz,
            double &mnf, uint &nxtHigh, double &mxf, uint &nxtLow, boost::mutex &m, SVMTrainer<T> *s)
        : ah(aH), al(aL), i(I), j(J), k(K), segSz(sgSz),
          minf(mnf), nextHigh( nxtHigh), maxf(mxf), nextLow( nxtLow), mtx( m), svm(s)
    {}   // end ctor

    void operator()()
    {
        const T &xi = svm->xs[i];
        const T &xj = svm->xs[j];
        const double aHigh = ah;
        const double aLow = al;

        const uint nxtSegIdx = k + segSz;
        const vector<T> &xs = svm->xs;
        KernelCache<T> *kernelCache = svm->kernelCache;
        vector<double> &fns = svm->fns;

        const unordered_set<uint> &highIdxs = svm->highIdxs;
        const unordered_set<uint> &lowIdxs = svm->lowIdxs;

        double mnf = INFINITY;
        uint nxtHigh = nextHigh;
        double mxf = -INFINITY;
        uint nxtLow = nextLow;

        // No mutex needed for the array since different objects work over different sections
        while ( k < nxtSegIdx)
        {
            if ( xs[k].size() != xi.size())
            {
                std::cerr << "xs[" << k << "].size() = " << xs[k].size() << " != xi.size() = " << xi.size() << std::endl;
                assert(false);
            }   // end if

            if ( xs[k].size() != xj.size())
            {
                std::cerr << "xs[" << k << "].size() = " << xs[k].size() << " != xj.size() = " << xj.size() << std::endl;
                assert(false);
            }   // end if

            fns[k] += aHigh*kernelCache->krn( i, xi, k, xs[k]) + aLow*kernelCache->krn( j, xj, k, xs[k]);

            // Test local high/low changes
            if ( highIdxs.count(k) && fns[k] < mnf)
            {
                mnf = fns[k];
                nxtHigh = k;
            }   // end if

            if ( lowIdxs.count(k) && fns[k] > mxf)
            {
                mxf = fns[k];
                nxtLow = k;
            }   // end if

            k++;
        }   // end while

        boost::mutex::scoped_lock lock( mtx);
        if ( mnf < minf)
        {
            minf = mnf;
            nextHigh = nxtHigh;
        }   // end if

        if ( mxf > maxf)
        {
            maxf = mxf;
            nextLow = nxtLow;
        }   // end if
    }   // end operator()

private:
    const double ah, al;
    const uint i, j;
    uint k;
    const uint segSz;

    double &minf;
    uint &nextHigh;
    double &maxf;
    uint &nextLow;
    boost::mutex &mtx;

    SVMTrainer<T> *svm;
};  // end class ThreadFn



template <typename T>
SVMTrainer<T>::SVMTrainer( const SVMParams &svmp, uint mt) throw (InvalidKernelException)
    : MAXTHREADS(mt), COST(svmp.cost()), EPS(svmp.eps()),
    kernel( svmp.makeKernel<T>()), kernelCache(NULL)
{
    if ( mt == 0)
        MAXTHREADS = boost::thread::hardware_concurrency();
}   // end ctor



template <typename T>
SVMTrainer<T>::SVMTrainer( const typename KernelFunc<T>::Ptr kf, double cost, double tolerance, uint mt)
    : MAXTHREADS(mt), COST(cost), EPS(tolerance), kernel(kf), kernelCache(NULL)
{
    if (mt == 0)
        MAXTHREADS = boost::thread::hardware_concurrency();
}   // end ctor


template <typename T>
SVMTrainer<T>::~SVMTrainer()
{
    if ( kernelCache != NULL) delete kernelCache;
}   // end dtor


template <typename T>
void SVMTrainer<T>::enableErrorOutput( bool enable)
{
    enableErrOut_ = enable;
}   // end enableErrorOutput


template <typename T>
SVMClassifier::Ptr SVMTrainer<T>::train( const vector<T> &pos, const vector<T> &neg)
{
    //static const uint SEC_ORD_HEUR_GAP = 1; // Every iteration
    static const uint SAMPLESIZE = 100;

    // For timing training
    struct timeval startTime;
    gettimeofday( &startTime, NULL);

    if ( pos.empty() || neg.empty())
    {
        SVMClassifier::Ptr null;
        return null;
    }   // end if

    reset( pos, neg);

    double bHigh = -1;
    double bLow = 1;

    Alpha ah( 0, this); // First positive example
    Alpha al( negZero, this);  // First negative example

    uint smpCnt = 0;
    if ( enableErrOut_)
    {
        cerr << "  B Max  |  B Min  |  (pair)" << endl;
        cerr << "===============================" << endl;
        cerr << std::setprecision(4) << std::fixed;
    }   // end if - ERROR OUTPUT

    while ( bLow - bHigh >= EPS)
    {
        optimise( ah, al, bHigh - bLow);
        updateIndexSets( ah.alpha, ah.idx);
        updateIndexSets( al.alpha, al.idx);

        // Updates functional predictions, updates bHigh and bLow, and computes
        // next working set alpha pair (ah and al) using first order heuristic.
        updatePredictions( ah, al, bHigh, bLow);

        if ( enableErrOut_)
        {
            // Select next low index based on second order heuristic. This is more complex
            // to calculate so is used less frequently than the first order heuristic 
            // already calculated at only constant cost.
            //if ( smpCnt % SEC_ORD_HEUR_GAP == 0)
            //    al.idx = selectSecondOrderPartner( ah.idx);

            if ( smpCnt++ % SAMPLESIZE == 0)
            {
                char ahc = target(ah.idx) == 1 ? '+' : '-';
                char alc = target(al.idx) == 1 ? '+' : '-';
                cerr << std::right << std::setw(8) << bHigh
                     << " | " << std::setw(7) << bLow
                     << " | " << ah.idx << "," << al.idx << " (" << ahc << alc << ")" << endl;
            }   // end if
        }   // end if - ERROR OUTPUT
    }   // end while

    if ( enableErrOut_)
    {
        cerr << "========== CONVERGED ==========" << endl;
        cerr << std::setprecision(4) << std::fixed;
        // Calculate time taken to train
        struct timeval endTime;
        gettimeofday( &endTime, NULL);
        uint msecs = (endTime.tv_sec - startTime.tv_sec) * 1000;
        msecs += (int)round((double)(endTime.tv_usec - startTime.tv_usec) * 0.001);
        cerr << " " << smpCnt << " iterations (" << msecs << " msecs)" << endl;
    }   // end if - ERROR OUTPUT

    delete kernelCache;
    kernelCache = NULL;
    return createClassifier( (bLow + bHigh)/2);
}   // end train


template <typename T>
double SVMTrainer<T>::constrainAlpha( double a)
{
    if ( a > COST - TAU) a = COST;
    else if ( a < TAU) a = 0;
    return a;
}   // end constrainAlpha


template <typename T>
void SVMTrainer<T>::optimise( Alpha &high, Alpha &low, double bDiff)
{
    const uint i = high.idx;
    const uint j = low.idx;
    const int yi = target(i);
    const int yj = target(j);
    const T &xi = xs[i];
    const T &xj = xs[j];
    const double kii = kernelCache->krn( i, xi, i, xi);
    const double kjj = kernelCache->krn( j, xj, j, xj);
    const double kij = kernelCache->krn( i, xi, j, xj);
    const double eta = kii + kjj - 2*kij;

    const double oja = alphas[j];
    const double oia = alphas[i];

    low.alpha = constrainAlpha( oja + yj*bDiff/eta);
    high.alpha = constrainAlpha( oia + yi*yj*(oja - low.alpha));
    low.alpha = constrainAlpha( oja + yi*yj*(oia - high.alpha));
}   // end optimise


template <typename T>
void SVMTrainer<T>::updatePredictions( Alpha &high, Alpha &low, double &bHigh, double &bLow)
{
    const uint i = high.idx;
    const uint j = low.idx;

    const double ah = (high.alpha - alphas[i]) * target(i);
    const double al = (low.alpha - alphas[j]) * target(j);

    uint nextHigh = i;
    uint nextLow = j;

    bHigh = INFINITY;
    bLow = -INFINITY;
    boost::mutex mtx;

    const uint fnsSz = fns.size();
    const uint SEGSIZE = fnsSz / MAXTHREADS;
    const uint REM = fnsSz % MAXTHREADS;
    uint segSz = SEGSIZE;
    uint segOffset = 0;
    boost::thread_group tGrp;
    for ( uint t = 0; t < MAXTHREADS; ++t)
    {
        segSz = SEGSIZE;
        if ( t < REM) segSz++;
        ThreadFn tFnObj( ah, al, i, j, segOffset, segSz, bHigh, nextHigh, bLow, nextLow, mtx, this);
        segOffset += segSz;
        boost::thread *tThrd = new boost::thread( tFnObj);
        tGrp.add_thread( tThrd);
    }   // end for
    tGrp.join_all();  // Member threads deleted at end of function

    // Update Alphas and set first order heuristic for next Alpha pair
    high.update( nextHigh);
    low.update( nextLow);
}   // end updatePredictions


template <typename T>   // Second order heuristic by Fan et al. 2005
uint SVMTrainer<T>::selectSecondOrderPartner( const uint i) const
{
    uint bestj = i; // Select j
    double objMin = -INFINITY;
    const double kii = kernelCache->krn( i, xs[i], i, xs[i]);

    BOOST_FOREACH( uint t, lowIdxs)
    {
        if ( fns[i] >= fns[t])
            continue;

        double eta = kii + kernelCache->krn( t, xs[t], t, xs[t])
                       - 2*kernelCache->krn( i, xs[i], t, xs[t]);
        if ( eta <= TAU)
            eta = TAU;

        double bdiff = fns[i] - fns[t];
        double deltaf = bdiff*bdiff/eta;
        if ( deltaf >= objMin)
        {
            objMin = deltaf;
            bestj = t;
        }   // end if
    }   // end foreach

    return bestj;
}   // end selectSecondOrderPartner


template <typename T>
void SVMTrainer<T>::updateIndexSets( const double a, const uint idx)
{
    int y = target( idx);
    lowIdxs.erase( idx);
    highIdxs.erase( idx);
    if (( a < COST && y == 1) || ( a > 0 && y == -1))
        highIdxs.insert( idx);
    if (( a > 0 && y == 1) || ( a < COST && y == -1))
        lowIdxs.insert( idx);
}   // end updateIndexSets


template <typename T>
SVMClassifier::Ptr SVMTrainer<T>::createClassifier( double threshold) const
{
    vector<double> *svAlphas = new vector<double>();
    vector<T> *svExamples = new vector<T>();
    for ( uint j = 0; j < xs.size(); ++j)
    {
        if ( alphas[j] <= TAU) continue;
        svAlphas->push_back( alphas[j] * target(j));
        svExamples->push_back( xs[j]);
    }   // end foreach

    SVMParams svmp( COST, EPS, kernel);
    SVMClassifier *cfier = new SVMClassifier(
            svmp, svAlphas, svExamples, threshold, true, negZero, xs.size() - negZero);
    return SVMClassifier::Ptr( cfier);
}   // end createClassifier


template <typename T>
void SVMTrainer<T>::reset( const vector<T> &pos, const vector<T> &neg)
{
    negZero = pos.size();   // Starting index of negative examples
    alphas.clear();
    fns.clear();
    xs.clear();

    uint i = 0;
    BOOST_FOREACH( const T &x, pos)
    {
        xs.push_back( x);
        alphas.push_back( 0);
        fns.push_back( -1);
        highIdxs.insert( i++);
    }   // end foreach

    BOOST_FOREACH( const T &x, neg)
    {
        xs.push_back( x);
        alphas.push_back( 0);
        fns.push_back( 1);
        lowIdxs.insert( i++);
    }   // end foreach

    kernelCache = new KernelCache<T>( kernel, xs.size());
}   // end reset


template <typename T>
int SVMTrainer<T>::target( uint idx) const
{
    return idx < negZero ? 1 : -1;
}   // end target


// static
template <typename T>
SVMClassifier::Ptr SVMTrainer<T>::train( const vector<T>& pos, const vector<T>& neg, const SVMParams& svmp)
{
    const int nthreads = boost::thread::hardware_concurrency();
    SVMTrainer<T> svmt( svmp, nthreads);
    return svmt.train( pos, neg);
}   // end train
