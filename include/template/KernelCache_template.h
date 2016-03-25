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
