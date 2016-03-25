#pragma once
#ifndef RLEARNING_HOG_MODEL_H
#define RLEARNING_HOG_MODEL_H

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include <iostream>
using std::ostream;
using std::istream;
#include "ObjModel.h"
using RLearning::ObjModel;
#include "ProHOGModel.h"
using RLearning::ProHOGModel;


namespace RLearning
{

class HOGModel : public ProHOGModel
{
public:
    static const string Type;   // "HOG"
    typedef boost::shared_ptr<HOGModel> Ptr;

    HOGModel(); // For operator>>
    HOGModel( const string &name,
              const cv::Size2f &objSz,
              const SVMClassifier::Ptr &svmc,
              bool dirDep=true,
              int cellSz=8);   // The square dimensions of each cell (in pixels)
    virtual ~HOGModel() {}

    inline int getCellPixels() const { return cellPxls;}
    virtual string getModelType() const;

protected:
    virtual void writeHeader( ostream&) const;
    // No need to override ProHOGModel::writeBody

    virtual string readHeader( istream&) throw (ObjModel::Exception);
    // No need to override ProHOGModel::readBody

    friend ostream &operator<<( ostream&, const HOGModel&);
    friend istream &operator>>( istream&, HOGModel&);

private:
    int cellPxls;
};  // end class


ostream &operator<<( ostream&, const HOGModel&);
istream &operator>>( istream&, HOGModel&);

}   // end namespace

#endif
