#pragma once
#ifndef RLEARNING_SVM_MODEL
#define RLEARNING_SVM_MODEL

#include "Model.h"
using RLearning::Model;
#include "SVMClassifier.h"
using RLearning::SVMClassifier;


namespace RLearning
{

class SVMModel : public Model
{
public:
    static const string Type;   // "SVM"
    typedef boost::shared_ptr<SVMModel> Ptr;

    SVMModel();  // For use with operator>>
    SVMModel( const string &name,      // Name of the object
              const cv::Size2f &objSz, // Real size of the object (as trained)
              const SVMClassifier::Ptr &svmc); // The trained classifier
    virtual ~SVMModel() {}

    virtual cv::Size getModelDims( int *nbins) const;
    virtual string getModelType() const;

    virtual double predict( const cv::Mat &z);

protected:
    virtual void writeHeader( ostream&) const;
    virtual void writeBody( ostream&) const;    // Writes out SVM byte data

    virtual string readHeader( istream&) throw (Model::Exception);
    virtual void readBody( istream&);   // Reads in SVM byte data

    friend ostream &operator<<( ostream&, const SVMModel&);
    friend istream &operator>>( istream&, SVMModel&);

private:
    SVMClassifier::Ptr svmc;    // The SVM classifier
};  // end class


ostream &operator<<( ostream&, const SVMModel&);
istream &operator>>( istream&, SVMModel&);

}   // end namespace

#endif
