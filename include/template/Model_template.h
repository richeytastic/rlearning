#include <sstream>
using std::istringstream;
using std::ostringstream;
#include <ios>


static void failAndThrow( istream &is, const string &msg) throw (Model::Exception)
{
    is.setstate( std::ios::failbit);
    throw Model::Exception( msg);
}   // end failAndThrow


template <typename T>
T Model::readToken( istream &is, const string &lab) throw (Model::Exception)
{
    string inLine;
    std::getline( is, inLine);
    istringstream iss(inLine);
    string inLab;
    iss >> inLab;
    string lb = lab;
    boost::algorithm::to_lower(lb); // Ensure provided label is lower case
    boost::algorithm::to_lower(inLab);  // Ensure read label is lower case
    if ( inLab.compare(lb) != 0)
    {
        ostringstream oss;
        oss << "Tried to read token " << lab
            << " into Model instance but found token " << inLab << " instead!";
        failAndThrow( is, oss.str());
    }   // end if

    T tok;

    try
    {
        iss >> tok;
    }   // end try
    catch ( const std::exception &e)
    {
        ostringstream oss;
        oss << "Tried and failed to read token " << lab << " into Model instance!";
        failAndThrow( is, oss.str());
    }   // end catch

    return tok;
}   // end readToken
