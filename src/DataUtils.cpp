#include "DataUtils.h"


void RLearning::generateRandomNormalData( boost::random::mt19937& mt,
        const cv::Mat &mn, const cv::Mat &stddev, int count, cv::Mat &drows)
{
    assert( mn.type() == stddev.type());
    assert( mn.depth() == CV_32F);
    assert( mn.size() == stddev.size());
    assert( mn.channels() == 1);
    assert( stddev.channels() == 1);

    const cv::Mat mnFlat = mn.reshape(1,1);
    const cv::Mat stdDevFlat = stddev.reshape(1,1);

    const float *mnRow = mnFlat.ptr<float>(0);
    const float *sdRow = stdDevFlat.ptr<float>(0);
    const int cols = mnFlat.cols;

    drows.create( count, cols, CV_32FC1);
    for ( int i = 0; i < count; ++i)
    {
        float *xRow = drows.ptr<float>(i);
        for ( int j = 0; j < cols; ++j)
        {
            boost::random::normal_distribution<> normDist( mnRow[j], sdRow[j]);
            xRow[j] = normDist( mt);
        }   // end for
    }   // end for
}   // end generateRandomNormalData



void RLearning::concatToMat( const vector<cv::Mat> &fvs, cv::Mat &xs, cv::Mat &labels, int &i, int lab)
{
    BOOST_FOREACH( const cv::Mat &x, fvs)
    {
        labels.at<int>(0,i) = lab;
        cv::Mat x1;
        x.convertTo( x1, CV_32F);
        const cv::Mat x2 = x1.reshape(1,1);
        x2.copyTo( xs.row(i));
        i++;
    }   // end foreach
}   // end concatToMat
