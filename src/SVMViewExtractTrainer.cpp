#include "SVMViewExtractTrainer.h"
using RLearning::SVMViewExtractTrainer;


// static
cv::Mat_<cv::Vec3b> SVMViewExtractTrainer::extractColourImage( const ViewExtract::Ptr v)
{
    const PointCloud::Ptr pc = v->getData();
    const int rows = pc->rows();
    const int cols = pc->cols();
    cv::Mat_<cv::Vec3b> img( rows, cols);
    double x, y, z; // Not used
    for ( int i = 0; i < rows; ++i)
    {
        for ( int j = 0; j < cols; ++j)
        {
            cv::Vec3b& pxl = img(i,j);
            pc->from( i, j, x, y, z, pxl[2], pxl[1], pxl[0]);
        }   // end for
    }   // end for
    return img;
}   // end extractColourImage



// static
cv::Mat_<float> SVMViewExtractTrainer::extractRangeImage( const ViewExtract::Ptr v)
{
    const PointCloud::Ptr pc = v->getData();
    const int rows = pc->rows();
    const int cols = pc->cols();
    cv::Mat_<float> img( rows, cols);
    float x, y; // Not used
    byte r,g,b; // Not used
    for ( int i = 0; i < rows; ++i)
    {
        for ( int j = 0; j < cols; ++j)
        {
            float& v = img(i,j);
            pc->from( i, j, x, y, v, r, g, b);
        }   // end for
    }   // end for
    return img;
}   // end extractRangeImage



// static
SVMViewExtractTrainer::Ptr SVMViewExtractTrainer::create( double cost, double eps)
{
    return Ptr( new SVMViewExtractTrainer( cost, eps));
}   // end create



SVMViewExtractTrainer::SVMViewExtractTrainer( double cost, double eps)
{
    svmp_ = SVMParams(cost, eps);   // Linear kernel by default
}   // end ctor



void SVMViewExtractTrainer::setCost( double cost)
{
    svmp_.cost(cost);
}   // end setCost



void SVMViewExtractTrainer::setConvergence( double eps)
{
    svmp_.eps(eps);
}   // end setConvergence



int SVMViewExtractTrainer::loadPositives( const string &dataDir)
{
    return loadData( dataDir, pdata_);
}   // end loadPositives



int SVMViewExtractTrainer::loadNegatives( const string &dataDir)
{
    return loadData( dataDir, ndata_);
}   // end loadNegatives



void extractColourData( const list<ViewExtract::Ptr>& data, vector<cv::Mat>& exs)
{
    using RFeatures::ImageGradientsBuilder;
    using RFeatures::IntegralImage;
    using RFeatures::ProHOG;

    // ProHOG params
    const cv::Size cellDims(7,4); // 7 cells wide by 4 high
    const int nbins = 9;    // Number of orientation bins for HOG

    exs.clear();
    BOOST_FOREACH( const ViewExtract::Ptr ve, data)
    {
        cv::Mat_<cv::Vec3b> img = SVMViewExtractTrainer::extractColourImage( ve);
        ImageGradientsBuilder::Ptr igb = ImageGradientsBuilder::create( img, nbins);
        IntegralImage<double>::Ptr pxlGrads = igb->getIntegralGradients();
        cv::Mat fimg = ProHOG( pxlGrads)( cellDims);
        exs.push_back( fimg);
    }   // end foreach
}   // end extractColourData



SVMClassifier::Ptr SVMViewExtractTrainer::trainOnValue()
{
    extractColourData( pdata_, pexs_);
    extractColourData( ndata_, nexs_);
    return train();
}   // end trainOnValue



void extractRangeData( const list<ViewExtract::Ptr>& data, vector<cv::Mat>& exs)
{
    using RFeatures::RangeGradientsBuilder;
    using RFeatures::IntegralImage;
    using RFeatures::ProHOG;

    // ProHOG params
    const cv::Size cellDims(7,4); // 7 cells wide by 4 high
    const int nbins = 9;    // Number of orientation bins for HOG

    exs.clear();
    BOOST_FOREACH( const ViewExtract::Ptr ve, data)
    {
        cv::Mat_<float> img = SVMViewExtractTrainer::extractRangeImage( ve);
        RangeGradientsBuilder::Ptr rgb = RangeGradientsBuilder::create( img, nbins);
        IntegralImage<double>::Ptr pxlGrads = rgb->getIntegralGradients();
        cv::Mat fimg = ProHOG( pxlGrads)( cellDims);
        exs.push_back( fimg);
    }   // end foreach
}   // end extractRangeData



SVMClassifier::Ptr SVMViewExtractTrainer::trainOnRange()
{
    extractRangeData( pdata_, pexs_);
    extractRangeData( ndata_, nexs_);
    return train();
}   // end trainOnRange



SVMClassifier::Ptr SVMViewExtractTrainer::train()
{
    const int maxThreads = boost::thread::hardware_concurrency();
    KernelFunc<cv::Mat>::Ptr kernel = svmp_.makeKernel<cv::Mat>();
    SVMTrainer<cv::Mat> svmTrainer( kernel, svmp_.cost(), svmp_.eps(), maxThreads);
    return svmTrainer.train( pexs_, nexs_);
}   // end train



int SVMViewExtractTrainer::loadData( const string& dataDir, list<ViewExtract::Ptr>& data)
{
    using std::cerr;
    using std::endl;
    using std::ifstream;

    try
    {
        data.clear();
        using namespace boost::filesystem;
        directory_iterator endItr;
        for ( directory_iterator itr(dataDir); itr != endItr; ++itr)
        {
            ifstream ifs( itr->path().c_str());
            ViewExtract::Ptr ve( new ViewExtract);
            ifs >> *ve;
            ifs.close();
            data.push_back(ve);
        }   // end for
    }   // end try
    catch ( const std::exception &e)
    {
        cerr << "ERROR: Failed to load training data in SVMViewExtractTrainer!" << endl;
        cerr << e.what() << endl;
        data.clear();
        return -1;
    }   // end catch

    return data.size();
}   // end loadData
