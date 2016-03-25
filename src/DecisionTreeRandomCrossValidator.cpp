#include "DecisionTreeRandomCrossValidator.h"
using RLearning::DecisionTreeRandomCrossValidator;


DecisionTreeRandomCrossValidator::DecisionTreeRandomCrossValidator( int tcount, int numIts,
            const cv::Mat &xs, const cv::Mat &labels, int numEVs)
    : RandomCrossValidator( tcount, numIts, xs, labels, numEVs)
{
}   // end ctor


void DecisionTreeRandomCrossValidator::train( const cv::Mat &trainData, const cv::Mat &labels)
{
    CvDTree *c = new CvDTree;
    c->train( trainData, CV_ROW_SAMPLE, labels.t());
    model_ = boost::shared_ptr<CvDTree>( c);
}   // end trainModel



float DecisionTreeRandomCrossValidator::validate( const cv::Mat &x)
{
    const CvDTreeNode *node = model_->predict(x);
    return node->value;
}   // end validate

