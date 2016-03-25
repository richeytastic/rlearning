/**
 * Wraps OpenCV ML models. Currently supported models:
 * CvKNearest
 */

#pragma once
#ifndef RLEARNING_CV_MODEL
#define RLEARNING_CV_MODEL

#include "Model.h"
using RLearning::Model;
#include <boost/shared_ptr.hpp>


namespace RLearning
{

class CvModel : public Model
{
public:
    CvModel();  // For use with operator>>
    CvModel( const string &name,      // Name of the object
             const cv::Size2f &objSz, // Real size of the object (as trained)
             const string &fvType,    // Type of feature e.g. "ProHOG"
             const string &modType,   // Class name of CV model e.g. "CvKNearest"
             const boost::shared_ptr<CvStatModel> model);
    virtual ~CvModel() {}

    virtual string getModelType() const;

    virtual double predict( const cv::Mat &z);

protected:
    virtual void writeHeader( ostream&) const;
    virtual void writeBody( ostream&) const;

    virtual string readHeader( istream&) throw (Model::Exception);
    virtual void readBody( istream&);

    friend ostream &operator<<( ostream&, const CvModel&);
    friend istream &operator>>( istream&, CvModel&);

private:
    string modType_;
    boost::shared_ptr<CvStatModel> model_;
};  // end class


ostream &operator<<( ostream&, const CvModel&);
istream &operator>>( istream&, CvModel&);

}   // end namespace

#endif
