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

#pragma once
#ifndef RLearning_SVM_PARAMS
#define RLearning_SVM_PARAMS

#include <string>
using std::string;
#include <iostream>
#include <sstream>
#include <exception>
#include <cassert>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>

#include "KernelFunc.h"
using RLearning::KernelFunc;


namespace RLearning
{

class InvalidKernelException : public std::exception
{
public:
    InvalidKernelException( const string &err) : m_err(err){}
    virtual ~InvalidKernelException() throw(){}
    virtual const char* what() const throw(){ return m_err.c_str();}
    virtual string error() const throw(){ return m_err;}
    virtual string errStr() const throw(){ return m_err;}
private:
    string m_err;
}; // end class


class SVMParams
{
public:
    // Define default SVM params with linear kernel.
    // Alternatively, this constructor can be used to read in
    // parameters with operator>>.
    SVMParams( double cost=1, double eps=1e-4);    // With linear kernel (or for reading in)

    // Define new SVM parameters with explicitly defined misclassification cost
    // and convergence parameter (eps). The kernel type string may be one of
    // "linear", "poly", "rbf" or "sigmoid". For the linear case, all other
    // parameters are ignored. For the other kernels, the respective functions
    // are defined as follows (with x1 and x2 representing training instances):
    // poly: pow( gam * x1.dot(x2) + cf0, deg)
    // rbf: let d = x1-x2, then exp( -gam * d.dot(d))
    // sigmoid: tanh( gam * x1.dot(x2) + cf0);
    SVMParams( double cost, double eps, const string &ktype, double gam=1, double cf0=0, double deg=1)
        throw (InvalidKernelException);

    template <typename T>
    SVMParams( double cost, double eps, const boost::shared_ptr<KernelFunc<T> >);

    template <typename T>
    void setKernelParams( const boost::shared_ptr<KernelFunc<T> >);

    template <typename T>
    boost::shared_ptr<KernelFunc<T> > makeKernel() const;

    // SVM training cost and convergence parameters
    inline double cost() const { return cost_;}
    inline double eps() const { return eps_;}
    void cost( double c) { cost_ = c;}
    void eps( double e) { eps_ = e;}

    // Kernel function parameters
    inline string kernel() const { return kernel_;}
    inline double gamma() const { return gamma_;}
    inline double coef0() const { return coef0_;}
    inline double degree() const { return degree_;}

    // Convenience functions for checking kernel type
    bool isLinear() const;
    bool isPoly() const;
    bool isRBF() const;
    bool isSigmoid() const;

    // Takes params as (with example):
    // Cost EPS Type Gamma Coef0 Degree
    // 1 1e-4 linear 1 0 1
    static SVMParams fromSpec( const std::string& svmspec);

    std::string toSpec() const;

private:
    double cost_;
    double eps_;
    string kernel_;
    double gamma_;
    double coef0_;
    double degree_;

    friend std::istream &operator>>( std::istream &is, SVMParams &p) throw (InvalidKernelException);

    static bool isKernelValid( const string &ktype);
    static bool isLinearKernelType( const string &ktype);
    static bool isPolyKernelType( const string &ktype);
    static bool isRBFKernelType( const string &ktype);
    static bool isSigmoidKernelType( const string &ktype);
};  // end class


std::ostream &operator<<( std::ostream &os, const SVMParams &p);
std::istream &operator>>( std::istream &is, SVMParams &p) throw (InvalidKernelException);


#include "template/SVMParams_template.h"

}   // end namespace

#endif

