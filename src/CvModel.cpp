#include "CvModel.h"
using RLearning::CvModel;
#include <boost/algorithm/string.hpp>
#include <sstream>
using std::ostringstream;
#include <fstream>
using std::ifstream;
using std::ofstream;
#include <cassert>
#include <iostream>
using std::endl;


CvModel::CvModel() : Model( "", cv::Size2f(0,0), "")
{
}   // end ctor



CvModel::CvModel( const string &nm, const cv::Size2f &ms, const string &fvType,
                  const string &mid, const boost::shared_ptr<CvStatModel> mod)
    : Model( nm, ms, fvType), modType_(mid), model_(mod)
{
    assert( mod != NULL);
}   // end ctor



string CvModel::getModelType() const
{
    return modType_;
}   // end getModelType



double CvModel::predict( const cv::Mat &z) 
{
    // Cast model_ to correct type and call predict
    double v = 0;
    if ( modType_ == "CvKNearest")
    {
        CvKNearest *mod = (CvKNearest*)model_.get();
        static const int k = 1;
        //cv::Mat results(1, z.rows, CV_32FC1);
        cv::Mat dist( z.rows, k, CV_32FC1);
        v = 2 * mod->find_nearest( z, k, /*&results*/0, 0, 0, &dist) - 1; // -1 or 1
        // Multiply by distance to the class
        v *= dist.at<float>(0,0);
    }   // end if
    else if ( modType_ == "CvSVM")
    {
        CvSVM *mod = (CvSVM*)model_.get();
        v = mod->predict( z, true);
    }   // end else if
    else
    {
        std::cerr << "ERROR: Unsupported classifier type!: " << modType_ << endl;
        assert( false);
    }   // end else

    return v;
}   // end predict



void CvModel::writeHeader( ostream &s) const
{
    Model::writeHeader(s);
}   // end writeHeader



void CvModel::writeBody( ostream &os) const
{
    assert( model_ != NULL);
    const char *tmpFile = "cvstatmodel.tmp";
    model_->save( tmpFile, modType_.c_str());
    ifstream ifs( tmpFile);
    string ln;
    while ( std::getline( ifs, ln))
        os << ln << endl;
    ifs.close();
}   // end writeBody



string CvModel::readHeader( istream &s) throw (Model::Exception)
{
    return Model::readHeader(s);
}   // end readHeader



void CvModel::readBody( istream &is)
{
    const char *tmpFile = "cvstatmodel.tmp";
    // Read into temp file first
    ofstream ofs( tmpFile);
    string ln;
    while ( std::getline( is, ln))
        ofs << ln << endl;
    ofs.close();
   
    assert( modType_ == "CvKNearest");  // Only CvKNearest supported currently
    model_ = boost::shared_ptr<CvKNearest>( new CvKNearest);
    model_->load( tmpFile, modType_.c_str());
}   // end readBody



ostream& RLearning::operator<<( ostream &s, const CvModel &mod)
{
    mod.writeHeader(s);
    mod.writeBody(s);
    return s;
}   // end operator<<



istream& RLearning::operator>>( istream &s, CvModel &mod)
{
    mod.readHeader(s);
    mod.readBody(s);
    return s;
}   // end operator>>

