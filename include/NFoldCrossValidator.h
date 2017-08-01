#ifndef RLEARNING_NFOLD_CROSS_VALIDATOR_H
#define RLEARNING_NFOLD_CROSS_VALIDATOR_H

#include "CrossValidator.h" // RLearning


namespace RLearning
{

class NFoldCrossValidator : public CrossValidator
{
public:
    NFoldCrossValidator( int nfolds, const cv::Mat_<float>& xs, const cv::Mat_<int>& labels, int numEVs=0);

    virtual ~NFoldCrossValidator(){}

    virtual void printResults( ostream &os) const;

    int getPositiveValSize() const { return pSegSz_;}
    int getNegativeValSize() const { return nSegSz_;}

protected:
    virtual int createTrainingMask( const cv::Mat_<int> &labs, const vector<int> &counts, char *mask);

    virtual bool moreIterations() const;

    //virtual void train( const cv::Mat_<float>& tdata, const cv::Mat_<int>& tlabels);

    //virtual float validate( const cv::Mat_<float>& x);

private:
    int nfolds_;             // Number of folds (N)
    int iter_;               // Iteration of N-fold cross validation
    int pSegSz_;             // Size of 1/Nth of the positive examples
    int nSegSz_;             // Size of 1/Nth of the negative examples

    void init( int nf);
};  // end class

}   // end namespace

#endif
