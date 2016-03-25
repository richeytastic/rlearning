#include "Model.h"
using RLearning::Model;

const string Model::INVALID_MODEL = "Unknown!";


Model::Model( const string &nm, const cv::Size2f &ms, const string &fvType)
    : name_(nm), objSz_( ms), fvType_(fvType)
{ }   // end ctor


Model::Model(){}  // Needed for static function readModelType below


void Model::writeHeader( ostream &os) const
{
    os << "OBJECT: " << name_ << std::endl;
    os << "OBJECT_WIDTH: " << objSz_.width << std::endl;
    os << "OBJECT_HEIGHT: " << objSz_.height << std::endl;
    os << "FEATURE: " << fvType_ << std::endl;
    os << "MODEL: " << this->getModelType() << std::endl;
}   // end write


string Model::readHeader( istream &is) throw (Model::Exception)
{
    name_ = Model::readToken<string>( is, "OBJECT:");
    double width = Model::readToken<double>( is, "OBJECT_WIDTH:");
    double height = Model::readToken<double>( is, "OBJECT_HEIGHT:");
    objSz_ = cv::Size2f( width, height);
    fvType_ = Model::readToken<string>( is, "FEATURE:");
    return Model::readToken<string>( is, "MODEL:");
}   // end readHeader


#include <fstream>
// static
string Model::readModelType( const string &fname)
{
    string modType = Model::INVALID_MODEL;
    class EmptyModel : public Model
    {
    public:
        virtual string getModelType() const { return "";}
        virtual double predict( const cv::Mat&) { return 0;}
    };  // end class

    EmptyModel tmp;
    try
    {
        std::ifstream ifs(fname.c_str());
        modType = tmp.readHeader( ifs);
        ifs.close();
    }   // end try
    catch ( const Model::Exception &e)
    {
        std::cerr << "Unable to read model type from file: " << fname << std::endl;
    }   // end catch
    catch ( const std::exception &e)
    {
        std::cerr << "Error reading model type from file: " << fname << std::endl;
    }   // end catch

    return modType;
}   // end readModelType
