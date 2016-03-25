#pragma once
#ifndef RLearning_KERNEL_CACHE
#define RLearning_KERNEL_CACHE

#include "SymmetricMatrix.h"
using rlib::SymmetricMatrix;

#include "SymmetricBitSet.h"
using rlib::SymmetricBitSet;

#include "KernelFunc.h"
using RLearning::KernelFunc;
#include <vector>
using std::vector;


namespace RLearning
{

template <typename T>
class KernelCache
{
public:
    KernelCache( const typename KernelFunc<T>::Ptr kernel, size_t sz);
    ~KernelCache();

    double krn( uint i, const T &xi, uint j, const T &xj);

    // Return the kernel function object used for this cache.
    inline typename KernelFunc<T>::Ptr getKernel() const { return kernel;}

private:
    // The kernel function (linear, polynomial, gaussian etc)
    const typename KernelFunc<T>::Ptr kernel;
    SymmetricMatrix<double> *vals;  // Cached kernel function values
    SymmetricBitSet *flags;        // Whether kernel function values have been cached or not
};  // end class KernelCache

#include "template/KernelCache_template.h"

}   // end namespace

#endif
