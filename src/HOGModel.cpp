#include "HOGModel.h"
using RLearning::HOGModel;


const string HOGModel::Type = "HOG";


HOGModel::HOGModel()
{ }   // end ctor


HOGModel::HOGModel( const string &nm, const cv::Size2f &ms,
                    const SVMClassifier::Ptr &s, bool dd, int cs)
    : ProHOGModel( nm, ms, s, dd), cellPxls(cs)
{ }   // end ctor


string HOGModel::getModelType() const
{
    return HOGModel::Type;
}   // end getModelType


void HOGModel::writeHeader( ostream &s) const
{
    ProHOGModel::writeHeader(s);
    s << "CELL_PIXELS: " << cellPxls << endl;
}   // end writeHeader


string HOGModel::readHeader( istream &s) throw (ObjModel::Exception)
{
    string typ = ProHOGModel::readHeader(s);
    cellPxls = ObjModel::readToken<int>( s, "CELL_PIXELS:");    // may throw
    return typ;
}   // end readHeader


ostream& RLearning::operator<<( ostream &s, const HOGModel &mod)
{
    mod.writeHeader(s);
    mod.writeBody(s);  // Not overridden here so calls ProHOGModel::writeBody
    return s;
}   // end operator<<


istream& RLearning::operator>>( istream &s, HOGModel &mod)
{
    mod.readHeader(s);
    mod.readBody(s);   // Not overridden here so calls ProHOGModel::readBody
    return s;
}   // end operator>>
