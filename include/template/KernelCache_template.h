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

template <typename T>
KernelCache<T>::KernelCache( const typename KernelFunc<T>::Ptr kf, size_t sz)
    : kernel(kf), vals( new SymmetricMatrix<double>( sz)), flags( new SymmetricBitSet( sz))
{}   // end ctor


template <typename T>
KernelCache<T>::~KernelCache()
{
    delete vals;
    delete flags;
}   // end dtor


template <typename T>
double KernelCache<T>::krn( uint i, const T &xi, uint j, const T &xj)
{
    if (flags->isSet( i,j)) // Return cached result if available
        return vals->get(i,j);
    double v = (*kernel)( xi, xj);  // Expensive...
    vals->set(i,j,v);   // ... so cache!
    flags->set(i,j);
    return v;
}   // end krn
