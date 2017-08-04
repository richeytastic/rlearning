/************************************************************************
 * Copyright (C) 2017 Richard Palmer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

#include <SVMClassifier.h>
using RLearning::SVMClassifier;
#include <cassert>
#include <iostream>
#include <fstream>
using std::ofstream;
using std::ifstream;
#include <sstream>
using std::stringstream;
using std::ios;
using std::cerr;
using std::endl;


// static
SVMClassifier::Ptr SVMClassifier::create()
{
    return SVMClassifier::Ptr( new SVMClassifier);
}   // end create



void SVMClassifier::deleteVectors()
{
    if ( delVecs && as != NULL)
    {
        delete as;
        as = NULL;
    }   // end if
    if ( delVecs && xs != NULL)
    {
        delete xs;
        xs = NULL;
    }   // end if
}   // end deleteVectors



SVMClassifier::SVMClassifier( const SVMParams &svmParams, vector<double> *alphas, vector<cv::Mat_<float> > *supportVectors,
                              double threshold, bool dvs, uint np, uint nn)
    : as( alphas), xs( supportVectors), // supportVectors are the training instances that define the hyperplane boundary
      b( threshold), numSVs( alphas->size()),
      delVecs(dvs), numPos(np), numNeg(nn)
{
    setKernel( svmParams);
    assert( numSVs == xs->size() && numSVs > 0);
    linx = cv::Mat_<float>::zeros( (*xs)[0].size());

    // If the kernel is linear, we can massively speed up classification by first adding
    // together all weighted examples so that classification itself only entails the scalar
    // product of this new "vector" with the test "vector".
    if ( svmp.isLinear())
    {
        for ( uint i = 0; i < numSVs; ++i)
        {
            // OpenCV throws a fit when you try and multiply a matrix by 1 or -1 using
            // Matrix expressions e.g. linx += m *alph. I HAVE NO IDEA WHY. So we use
            // cv::scaleAdd instead which seems to work just fine.
            //double alph = (*as)[i];
            //cv::Mat m = (*xs)[i];
            //linx += m * alph;
            cv::scaleAdd( (*xs)[i], (*as)[i], linx, linx);
        }   // end for

        deleteVectors();    // Don't need training data to use the linear classifier (because the support vectors give all the info needed)
    }   // end if
}   // end ctor



SVMClassifier::~SVMClassifier()
{
    deleteVectors();
}   // end dtor


class KernelThreadFunc
{
public:
    KernelThreadFunc( const vector<double>* as, const KernelFunc<cv::Mat_<float> >::Ptr krn, const vector<cv::Mat_<float> >* xs)
        : _as(as), _krn(krn), _xs(xs) {}

    void calcResult( int soff, int ssz, const cv::Mat_<float>* z)
    {
        double res = 0;
        const int segMax = soff + ssz;
        for ( int i = soff; i < segMax; ++i)
            res += _as->at(i) * (*_krn)( *z, _xs->at(i));
        _result = res;
    }   // end operator()

    double getResult() const { return _result;}

private:
    const vector<double>* _as;
    const KernelFunc<cv::Mat_<float> >::Ptr _krn;
    const vector<cv::Mat_<float> >* _xs;
    double _result;
};  // end class 



float SVMClassifier::predict( const cv::Mat_<float>& z) const
{
    if ( svmp.isLinear())
        return (z.dot(linx) - b)/linx.total();  // Normalise by the vector length

    double result = -b;

    KernelThreadFunc ktf( as, kernel, xs);
    ktf.calcResult( 0, numSVs, &z);
    result += ktf.getResult();
    return result / z.total();

    /*
    static const int nthreads = boost::thread::hardware_concurrency();   // CPU threads available
    const int segSz = numSVs / nthreads;
    int rem = numSVs % nthreads;

    boost::thread_group tgroup;
    vector<KernelThreadFunc*> tobjs(nthreads);
    int segOffset = 0;
    for ( int i = 0; i < nthreads; ++i)
    {
        int ssz = segSz;
        if ( rem > 0)
        {
            ssz++;
            rem--;
        }   // end if

        tobjs[i] = new KernelThreadFunc( as, kernel, xs);
        tgroup.create_thread( boost::bind( &KernelThreadFunc::calcResult, tobjs[i], segOffset, ssz, &z));  // Start thread
        segOffset += ssz; // Offset for next thread
    }   // end for
    tgroup.join_all();    // All processing done

    // Collect the result for return
    for ( int i = 0; i < nthreads; ++i)
    {
        result += tobjs[i]->getResult();
        delete tobjs[i];
    }   // end for

    return result / z.total();   // Normalise by the vector length
    */
}   // end predict



cv::Size SVMClassifier::getModelDims( int *channels) const
{
    if ( channels != NULL)
        *channels = linx.channels();
    return linx.size();
}   // end getModelDims



ostream& RLearning::operator<<( ostream &os, const SVMClassifier &svmc)
{
    os << svmc.svmp;
    os << "THRESHOLD: " << svmc.b << endl;
    os << "NUM_POS: " << svmc.numPos << endl;
    os << "NUM_NEG: " << svmc.numNeg << endl;
    os << "NUM_SVS: " << svmc.numSVs << endl;
    if ( svmc.svmp.isLinear())
    {
        RFeatures::writeBinary( os, svmc.linx);
#ifndef NDEBUG
        for ( int i = 0; i < svmc.linx.cols; ++i)
            std::cerr << i << ": " << ((float*)svmc.linx.ptr(0))[i] << std::endl;
#endif
    }   // end if
    else
    {
        for ( uint i = 0; i < svmc.numSVs; ++i)
        {
            os.write( (const char*)&(*svmc.as)[i], sizeof(double));  // Example weight
            RFeatures::writeBinary( os, (*svmc.xs)[i]);    // The example itself
        }   // end for
    }   // end else

    return os;
}   // end operator<<



// private
void SVMClassifier::setKernel( const SVMParams &p)
{
    svmp = p;
    kernel = p.makeKernel<cv::Mat_<float> >();
}   // end setKernel



istream& RLearning::operator>>( istream &is, SVMClassifier &svmc)
{
    // Read in preliminary details of classifier
    SVMParams svmp;
    is >> svmp;
    svmc.setKernel( svmp);
    string ln, lab;
    while ( getline( is, ln))
    {
        stringstream ss(ln);
        ss >> lab;
        if ( lab == "THRESHOLD:")
            ss >> svmc.b;
        else if ( lab == "NUM_POS:")
            ss >> svmc.numPos;
        else if ( lab == "NUM_NEG:")
            ss >> svmc.numNeg;
        else if ( lab == "NUM_SVS:")
        {
            ss >> svmc.numSVs;  // Support vector weights and instances follow
            break;  // Immediately after the specification of the number of SVs comes the SV data itself.
        }   // end else if
        else
        {
            cerr << "Unknown token on SVMClassifier read!" << endl;
            is.setstate(ios::failbit);
        }   // end else
    }   // end while

    if ( !is.good())
        return is;

    svmc.delVecs = false;
    svmc.as = NULL;
    svmc.xs = NULL;
    if ( is.good() && svmp.isLinear())
    {
        cv::Mat lnx;
        RFeatures::readBinary( is, lnx);
        assert( lnx.rows == 1);
        assert( lnx.type() == CV_32FC1);
#ifndef NDEBUG
        std::cerr << lnx.size() << std::endl;
        for ( int i = 0; i < lnx.cols; ++i)
            std::cerr << i << ": " << ((float*)lnx.ptr(0))[i] << std::endl;
#endif
        svmc.linx = cv::Mat_<float>(lnx);
    }   // end if
    else if ( is.good())
    {
        svmc.delVecs = true;
        svmc.as = new vector<double>;
        svmc.xs = new vector<cv::Mat_<float> >;
        for ( uint i = 0; i < svmc.numSVs; ++i)
        {
            double a;   // The example weight
            is.read( (char*)&a, sizeof(double));
            if ( !is.good())
                break;
            svmc.as->push_back(a);
            cv::Mat m;  // The example itself
            RFeatures::readBinary( is, m);
            assert( m.type() == CV_32FC1);
            if ( !is.good())
                break;
            svmc.xs->push_back((cv::Mat_<float>)m);
        }   // end for
    }   // end else

    return is;
}   // end operator>>
