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

/**
 * Defines common kernel functions for SVMs including:
 * Linear, Polynomial, Gaussian and Sigmoid
 *
 * Richard Palmer
 * April 2012
 */

#pragma once
#ifndef RLEARNING_KERNEL_FUNC
#define RLEARNING_KERNEL_FUNC

#include <cmath>
#include <string>
using std::string;
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>


namespace RLearning
{


template<typename T>
class KernelFunc
{
public:
    typedef boost::shared_ptr<KernelFunc<T> > Ptr;

    virtual double operator()( const T &x1, const T &x2) const = 0;  // The kernel function
    virtual string getType() const = 0; // Return string identifier of kernel type
};  // end class KernelFunc



template<typename T>
class LinearKernel : public KernelFunc<T>
{
public:
    virtual double operator()( const T &x1, const T &x2) const
    {
        return x1.dot(x2);
    }   // end operator()

    static string Type;
    virtual string getType() const { return LinearKernel::Type;}
};  // end class LinearKernel
template <typename T> string LinearKernel<T>::Type = "linear";



template<typename T>
class PolyKernel : public KernelFunc<T>
{
public:
    PolyKernel() : gam(1), cf0(0), degree(1) {}
    PolyKernel( double g, double c, double d) : gam(g), cf0(c), degree(d) {}

    virtual double operator()( const T &x1, const T &x2) const
    {
        return pow(gam * x1.dot(x2) + cf0, degree);
    }   // end operator()

    static string Type;
    virtual string getType() const { return PolyKernel::Type;}

    void setGamma( double g) { gam = g;}
    void setTerm( double c) { cf0 = c;}
    void setDegree( double d) { degree = d;}

    inline double getGamma() const { return gam;}
    inline double getTerm() const { return cf0;}
    inline double getDegree() const { return degree;}

private:
    double gam, cf0, degree;
};  // end class PolyKernel
template<typename T> string PolyKernel<T>::Type = "poly";



// Otherwise known as Radial Basis Function (RBF)
template<typename T>
class GaussianKernel : public KernelFunc<T>
{
public:
    GaussianKernel() : gam(1) {}
    GaussianKernel( double g) : gam(g) {}
    virtual double operator()( const T &x1, const T &x2) const
    {
        T d = x1 - x2;  // We assume that T implements dot() here!
        return exp( -gam*d.dot(d));
    }   // end operator()

    static string Type;
    virtual string getType() const { return GaussianKernel::Type;}

    void setGamma( double g) { gam = g;}
    inline double getGamma() const { return gam;}

private:
    double gam;
};  // end class GaussianKernel
template<typename T> string GaussianKernel<T>::Type = "rbf";



template<typename T>
class SigmoidKernel : public KernelFunc<T>
{
public:
    SigmoidKernel() : gam(1), cf0(0) {}
    SigmoidKernel( double g, double c) : gam(g), cf0(c) {}
    virtual double operator()( const T &x1, const T &x2) const
    {
        return tanh( gam*x1.dot(x2) + cf0);
    }   // end operator()

    static string Type;
    virtual string getType() const { return SigmoidKernel::Type;}

    void setGamma( double g) { gam = g;}
    void setTerm( double c) { cf0 = c;}
    inline double getGamma() const { return gam;}
    inline double getTerm() const { return cf0;}

private:
    double gam, cf0;
};  // end class SigmoidKernel
template<typename T> string SigmoidKernel<T>::Type = "sigmoid";


}   // end namespace


#endif
