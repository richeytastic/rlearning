#include "SVMDataMiner.h"
using RLearning::SVMDataMiner;

#include <cassert>
#include <cstdlib>
#include <iostream>

#define MIN_NEG_THRESH -0.05   // Threshold for hard negatives


namespace
{


// Get a value in range [min,max)
int rangeRand( int min, int max)
{
    const double rng = max - min - 1;
    return min + (int)(((double)random() / RAND_MAX) * rng + 0.5);
}   // end rangeRand



// Create and return a Pro-HOG feature vector from a random
// sub-region (no smaller than minSz) of the provided image.
// Parameter cellDims gives the required cell dimensions for
// the extracted Pro-HOG feature vector.
// Assumes that img has dimensions >= as given by minSz.
cv::Mat createNewRandomProHogFV( const cv::Mat &img, const cv::Size &minSz, const cv::Size &cellDims)
{
    assert( img.cols >= minSz.width);
    assert( img.rows >= minSz.height);

    // Define the random rectangle
    cv::Rect rct( rangeRand( 0, img.cols - minSz.width),
                  rangeRand( 0, img.rows - minSz.height), 0,0);
    rct.width = rangeRand( minSz.width, img.cols - rct.x);
    rct.height = rangeRand( minSz.height, img.rows - rct.y);

    return ProHOG( img)( cellDims, rct); // Extract and return Pro-HOG feature vector
}   // end createNewRandomProHogFV




// Finds hard negatives from random samples of negative images for a given classifier.
// Hard negatives are defined as either false positives or correctly classified negatives
// but very close to the classification boundary. The classification boundary is zero,
// but setting it to a negative value close to zero (constructor parameter minThresh)
// determines how close a correctly classified negative instance must be from the boundary
// to be labelled as a hard negative.
// If no classifier is passed in, a set of random feature vectors will be produced.
class HardNegsMiner
{
public:
    HardNegsMiner( const vector<cv::Mat> &nimgs,   // The selection of negative images
                   vector<cv::Mat> *hardNegs,      // Collated hard negatives
                   int minHardNegsReq,             // Minimum size of hardNegs before return
                   const cv::Size &cellDims,       // Pro-HOG feature extraction parameter
                   const SVMClassifier::Ptr svmc=SVMClassifier::Ptr(),  // The classifier to use for mining
                   double minThresh=MIN_NEG_THRESH,    // The minimum threshold for a "hard negative"
                   boost::mutex *mtx=NULL)         // Mutex for adding to hardNegs (if required)
    : nimgs_(nimgs), hardNegs_(hardNegs), minReq_(minHardNegsReq), cellDims_(cellDims),
      svmc_(svmc), minThresh_(minThresh), mtx_(mtx), hitRate_(0)
    {}    // end ctor


    void mine()
    {
        const int negAllCount = nimgs_.size();
        const cv::Size minSz(20,20);

        int sz = 0;
        int cnt = 0;
        int hit = 0;
        while ( sz < minReq_)
        {
            // Retrieve a random negative image
            int nidx = rangeRand( 0, negAllCount);
            cv::Mat negImg = nimgs_[nidx];
            // Ensure image is not too small for feature vector extraction
            if ( negImg.cols < minSz.width || negImg.rows < minSz.height)
                continue;

            cv::Mat fv = createNewRandomProHogFV( negImg, minSz, cellDims_);
            cnt++;
            if ( svmc_ == NULL || ( svmc_->predict( fv) >= minThresh_))
            {
                boost::mutex::scoped_lock lock( *mtx_);
                hardNegs_->push_back( fv);
                sz = hardNegs_->size();
                hit++;
            }   // end if
        }   // end while

        hitRate_ = 0;
        if ( cnt > 0)
            hitRate_ = (double)hit/cnt;
    }   // end mine


    // The proportion of samples classified as hard negatives.
    // Indicates the potential false positive rate of the classifier (as measured on the training data).
    inline double getHardNegsProportion() const { return hitRate_;}

private:
    const vector<cv::Mat> nimgs_;
    vector<cv::Mat> *hardNegs_;
    int minReq_;
    const cv::Size cellDims_;
    const SVMClassifier::Ptr svmc_;
    const double minThresh_;
    boost::mutex *mtx_;
    double hitRate_;
};  // end class



// Returns an estimate of the proportion of random negative samples that
// were identified as hard negatives given the provided classifier.
double mineHardNegatives_mt( const vector<cv::Mat> &nimgs,   // The negative images
                          vector<cv::Mat> &hardNegs,      // The hard negatives for return
                          int minHardNegsReq,             // Min number of hard negatives required
                          const cv::Size &cellDims,       // Pro-HOG feature extraction parameter
                          const SVMClassifier::Ptr svmc=SVMClassifier::Ptr(),
                          double minThresh=MIN_NEG_THRESH)
{
    boost::thread_group thrds;
    boost::mutex mtx; // Mutex for adding hard negs
    vector<HardNegsMiner*> miners;

    const int numThreads = boost::thread::hardware_concurrency();
    for ( int i = 0; i < numThreads; ++i)
    {
        HardNegsMiner *m = new HardNegsMiner( nimgs, &hardNegs, minHardNegsReq, cellDims, svmc, minThresh, &mtx);
        miners.push_back(m);
        thrds.create_thread( boost::bind( &HardNegsMiner::mine, m));
    }   // end for

    thrds.join_all();

    double avgHitRate = 0;
    BOOST_FOREACH( HardNegsMiner* m, miners)
    {
        avgHitRate += m->getHardNegsProportion();
        delete m;
    }   // end foreach

    avgHitRate /= miners.size();
    return avgHitRate;
}   // end mineHardNegatives_mt

}   // end namespace



SVMDataMiner::SVMDataMiner( const string &posDir, const string &negDir,
                            const cv::Size &csz, double cost, double eps, int maxIts)
    : cellDims_(csz), cost_(cost), eps_(eps), maxIterations_(maxIts)
{
    using namespace boost::filesystem;

    directory_iterator end_itr;

    assert( is_directory( path(posDir)));
    assert( is_directory( path(negDir)));

    // Load all the positive images
    std::cerr << "Loading all positive images... ";
    vector<cv::Mat> posImgs;
    for ( directory_iterator itr( posDir); itr != end_itr; ++itr)
    {
        assert( is_regular_file( itr->status()));
        cv::Mat img;
        if ( RFeatures::loadImage( itr->path().string(), img)) // Absolute path to file
        {
            cv::Mat img2;
            cv::flip( img, img2, 1);    // Flip about vertical axis
            posImgs.push_back( img);
            posImgs.push_back( img2);
        }   // end if
    }   // end for
    std::cerr << "(2 x " << posImgs.size()/2 << " flipped images)" << std::endl;

    // Load all the negative images
    std::cerr << "Loading all negative images... ";
    for ( directory_iterator itr( negDir); itr != end_itr; ++itr)
    {
        assert( is_regular_file( itr->status()));
        cv::Mat img;
        if ( RFeatures::loadImage( itr->path().string(), img)) // Absolute path to file
            negImgs_.push_back( img);
    }   // end for
    std::cerr << "(" << negImgs_.size() << " images)" << std::endl;

    std::cerr << "Extracting Pro-HOG feature vectors from positive instances..." << std::endl;
    RFeatures::BatchProHOGExtractor phExtractor( posImgs, 9, true, cellDims_);
    phExtractor.extract_mt( posInstances_);

    srandom(1);
}   // end ctor



void shrinkNegCache( const SVMClassifier::Ptr svmc, vector<cv::Mat> &cache)
{
    const int cacheSize = cache.size();
    for ( int i = 0; i < cacheSize; ++i)
    {
        cv::Mat x = cache.front();
        cache.erase( cache.begin());
        if ( svmc->predict( x) >= MIN_NEG_THRESH)
            cache.push_back( x);    // Return to end
    }   // end foreach
}   // end shrinkNegCache



void shrinkPosCache( const SVMClassifier::Ptr svmc, vector<cv::Mat> &cache)
{
    const int cacheSize = cache.size();
    for ( int i = 0; i < cacheSize; ++i)
    {
        cv::Mat x = cache.front();
        cache.erase( cache.begin());
        if ( svmc->predict( x) < -MIN_NEG_THRESH)
            cache.push_back( x);    // Return to end
    }   // end foreach
}   // end shrinkPosCache




SVMClassifier::Ptr SVMDataMiner::train()
{
    using std::cerr;
    using std::endl;

    // Cache of positive and negative examples for training
    vector<cv::Mat> posCache = posInstances_;
    vector<cv::Mat> negCache;

    SVMClassifier::Ptr svmc;
    const int maxThreads = boost::thread::hardware_concurrency();   // For SVMTrainer
    KernelFunc<cv::Mat>::Ptr kernel( new LinearKernel<cv::Mat>);

    int negLimit = posInstances_.size(); // Required number of negatives
    int iter = 0;

    vector<cv::Mat> oldNegs;
    while ( iter++ < maxIterations_)
    {
        cerr << iter << ") Mining hard negatives from random sub-regions of negative images..." << endl;
        // Grow negCache further from misclassifications of random negatives using the current classifier
        mineHardNegatives_mt( negImgs_, negCache, negLimit, cellDims_, svmc);

        cerr << "\tTraining cache sizes (pos,neg) = " << posCache.size() << ", " << negCache.size() << endl;

        SVMTrainer<cv::Mat> svmt( kernel, cost_, eps_, maxThreads);
        svmt.enableErrorOutput(false);
        svmc = svmt.train( posCache, negCache);

        // shrink positive and negative caches to the misclassified examples or those that are close to the margin
        shrinkNegCache( svmc, negCache);
        shrinkPosCache( svmc, posCache);
    }   // end while

    return svmc;
}   // end train



/*
SVMClassifier::Ptr SVMDataMiner::train()
{
    using std::cerr;
    using std::endl;

    // Cache of positive and negative examples for training
    vector<cv::Mat> posCache = posInstances_;
    vector<cv::Mat> negCache;

    SVMClassifier::Ptr svmc;
    const int maxThreads = boost::thread::hardware_concurrency();   // For SVMTrainer
    KernelFunc<cv::Mat>::Ptr kernel( new LinearKernel<cv::Mat>);

    int negLimit = posInstances_.size(); // Required number of negatives

    int iter = 0;
    //cv::Mat wOld; // Linear trained support vector weights to check for convergence

    vector<cv::Mat> oldNegs;

    while ( iter++ < maxIterations_)
    {
        cerr << iter << ") Mining hard negatives from random sub-regions of negative images..." << endl;
        // Grow negCache further from misclassifications of random negatives using the current classifier
        mineHardNegatives_mt( negImgs_, negCache, negLimit, cellDims_, svmc);

        cerr << "\tTraining cache sizes (pos,neg) = " << posCache.size() << ", " << negCache.size() << endl;

        SVMTrainer<cv::Mat> svmt( kernel, cost_, eps_, maxThreads);
        svmt.enableErrorOutput(false);
        svmc = svmt.train( posCache, negCache);

        // Convergence checking against previous classifier
        cv::Mat wNew = svmc->getLinearWeightsImg();
        double sDiff = -1;
        if ( iter == 1)
            wOld = wNew;
        else
        {   // Check for convergence (check sum of square diffs in this weights vs old weights)
            cv::Mat wDiff = wNew - wOld;
            sDiff = wDiff.dot(wDiff);
            cerr << "\tModel convergence value (sum of squares delta) = " << sDiff << endl;
            if (sDiff < eps_)
            {
                cerr << "\tCONVERGED!" << endl;
                break;  // Converged!
            }   // end if
        }   // end else
        /

        const int negCacheSize = negCache.size();

        cerr << "\tExpanding negatives cache to include hard negatives identified from old negatives...";
        // Grow cache of hard-negatives (false positives) from old random negatives
        const int sz = oldNegs.size();
        for ( int i = 0; i < sz; ++i)
        {
            cv::Mat nx = oldNegs.front();
            oldNegs.erase( oldNegs.begin());

            if ( svmc->predict( nx) >= MIN_NEG_THRESH)
                negCache.push_back( nx);
            else
                oldNegs.push_back(nx);
        }   // end foreach
        cerr << " (" << negCache.size() << ")" << endl;

        cerr << "\tShrinking negatives cache to only hard negatives...";
        // Shrink cache by keeping for the next round only the hard negative instances
        // from the current negative cache. Other negative examples are held over in oldNegs
        // for testing again in the next iteration.
        for ( int i = 0; i < negCacheSize; ++i) // Front of neg cache
        {
            cv::Mat nx = negCache.front();
            negCache.erase( negCache.begin());

            if ( svmc->predict( nx) >= MIN_NEG_THRESH)
                negCache.push_back( nx);    // Return to end
            else
                oldNegs.push_back(nx); // Not currently a false positive
        }   // end foreach
        cerr << " (" << negCache.size() << ")" << endl;

        cerr << "\tMining false negatives from all positive instances...";
        // New positive cache will consist of the misclassified positives according to the current classifier
        vector<cv::Mat> hardPos;
        BOOST_FOREACH( cv::Mat px, posInstances_)
        {
            // False-negatives == hard-positives
            if ( svmc->predict( px) < -MIN_NEG_THRESH)
                hardPos.push_back( px);
        }   // end foreach
        cerr << " (" << hardPos.size() << ")" << endl;
        posCache = hardPos;
        posCache = posInstances_;
        /
    }   // end while

    return svmc;
}   // end train
*/
