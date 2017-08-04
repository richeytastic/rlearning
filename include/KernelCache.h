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
