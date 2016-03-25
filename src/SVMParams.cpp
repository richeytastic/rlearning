#include "SVMParams.h"
using RLearning::SVMParams;
#include <sstream>
#include <boost/algorithm/string.hpp>


SVMParams::SVMParams( double cost, double eps)
    : cost_(cost), eps_(eps), kernel_(LinearKernel<cv::Mat>::Type), gamma_(1), coef0_(0), degree_(1)
{
}   // end ctor



SVMParams::SVMParams( double cost, double eps, const string &ktype, double gam, double cf0, double deg)
    throw (InvalidKernelException)
    : cost_(cost), eps_(eps), kernel_(ktype), gamma_(gam), coef0_(cf0), degree_(deg)
{
    if ( !SVMParams::isKernelValid( ktype))
    {
        std::ostringstream oss;
        oss << "ERROR: Invalid kernel type \"" << kernel_ << "\" encountered in SVMParams ctor!" << std::endl;
        throw InvalidKernelException( oss.str());
    }   // end if
}   // end ctor


bool SVMParams::isLinear() const
{
    return SVMParams::isLinearKernelType( kernel_);
}   // end isLinear


bool SVMParams::isPoly() const
{
    return SVMParams::isPolyKernelType( kernel_);
}   // end isPoly


bool SVMParams::isRBF() const
{
    return SVMParams::isRBFKernelType( kernel_);
}   // end isRBF


bool SVMParams::isSigmoid() const
{
    return SVMParams::isSigmoidKernelType( kernel_);
}   // end isSigmoid



// static
bool SVMParams::isKernelValid( const string &ktype)
{
    if ( isLinearKernelType( ktype))
        return true;
    if ( isPolyKernelType( ktype))
        return true;
    if ( isRBFKernelType( ktype))
        return true;
    if ( isSigmoidKernelType( ktype))
        return true;

    return false;
}   // end isKernelValid


// static
SVMParams SVMParams::fromSpec( const string& svmcfg)
{
    std::istringstream iss(svmcfg);
    double cost, eps;
    string ktype;
    double gam = 1;
    double cf0 = 0;
    double deg = 1;
    iss >> cost >> eps >> ktype >> gam >> cf0 >> deg;
    return SVMParams( cost, eps, ktype, gam, cf0, deg);
}   // end fromSpec


std::string SVMParams::toSpec() const
{
    std::ostringstream oss;
    oss << cost() << " " << eps() << " " << kernel() << " " << gamma() << " " << coef0() << " " << degree();
    return oss.str();
}   // end toSpec


bool isKernelType( const string &ktype, const string kcmp)
{
    string klab = ktype;
    boost::algorithm::to_lower( klab);
    if ( ktype.compare( kcmp) == 0)
        return true;
    return false;
}   // end isKernelType


// static
bool SVMParams::isLinearKernelType( const string &ktype)
{
    return isKernelType( ktype, LinearKernel<cv::Mat>::Type);
}   // end isLinearKernelType



// static
bool SVMParams::isPolyKernelType( const string &ktype)
{
    return isKernelType( ktype, PolyKernel<cv::Mat>::Type);
}   // end isPolyKernelType



// static
bool SVMParams::isRBFKernelType( const string &ktype)
{
    return isKernelType( ktype, GaussianKernel<cv::Mat>::Type);
}   // end isRBFKernelType



// static
bool SVMParams::isSigmoidKernelType( const string &ktype)
{
    return isKernelType( ktype, SigmoidKernel<cv::Mat>::Type);
}   // end isSigmoidKernelType



std::ostream &RLearning::operator<<( std::ostream &os, const SVMParams &p)
{
    using std::endl;
    os << "SVM_COST: " << p.cost() << endl;
    os << "SVM_EPS: " << p.eps() << endl;
    os << "KERNEL: " << p.kernel() << endl;
    os << "GAMMA: " << p.gamma() << endl;
    os << "COEF0: " << p.coef0() << endl;
    os << "DEGREE: " << p.degree() << endl;
    return os;
}   // end operator<<



std::istream &RLearning::operator>>( std::istream &is, SVMParams &p) throw (InvalidKernelException)
{
    using std::cerr;
    using std::endl;

    string ln, lab;
    while ( std::getline( is, ln))
    {
        if ( ln.empty())
            continue;
        std::istringstream iss(ln);
        iss >> lab;
        boost::algorithm::to_lower( lab); // Convert to lower case (in-place)

        if ( lab.compare( "svm_cost:") == 0)
            iss >> p.cost_;
        else if ( lab.compare( "svm_eps:") == 0)
            iss >> p.eps_;
        else if ( lab.compare( "kernel:") == 0)
        {
            iss >> p.kernel_;
            // Check validity of the kernel type
            if ( !SVMParams::isKernelValid( p.kernel_))
            {
                is.setstate(std::ios::failbit);
                std::ostringstream oss;
                oss << "ERROR: Invalid kernel type \"" << p.kernel_
                    << "\" encountered while reading SVM parameters!" << endl;
                throw InvalidKernelException( oss.str());
            }   // end if
        }   // end else if
        else if ( lab.compare( "gamma:") == 0)
            iss >> p.gamma_;
        else if ( lab.compare( "coef0:") == 0)
            iss >> p.coef0_;
        else if ( lab.compare( "degree:") == 0)
        {
            iss >> p.degree_;
            break;  // done
        }   // end else if
        else
        {
            cerr << "Invalid token reading in SVM parameters!: \"" << lab << "\"" << endl;
            is.setstate(std::ios::failbit);
        }   // end else
    }   // end while

    return is;
}   // end operator>>
