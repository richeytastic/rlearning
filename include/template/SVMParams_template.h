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
SVMParams::SVMParams( double cost, double eps, const boost::shared_ptr<KernelFunc<T> > k)
    : cost_(cost), eps_(eps), kernel_(LinearKernel<T>::Type), gamma_(1), coef0_(0), degree_(1)
{
    setKernelParams(k);
}   // end ctor


    
template <typename T>
void SVMParams::setKernelParams( const boost::shared_ptr<KernelFunc<T> > k)
{
    kernel_ = k->getType();
    gamma_ = 1;
    coef0_ = 0;
    degree_ = 1;

    if ( isPolyKernelType( kernel_))
    {
        PolyKernel<T> *pk = (PolyKernel<T>*)k.get();
        gamma_ = pk->getGamma();
        coef0_ = pk->getTerm();
        degree_ = pk->getDegree();
    }   // end if
    else if ( isRBFKernelType( kernel_))
    {
        GaussianKernel<T> *gk = (GaussianKernel<T>*)k.get();
        gamma_ = gk->getGamma();
    }   // end else if
    else if ( isSigmoidKernelType( kernel_))
    {
        SigmoidKernel<T> *sk = (SigmoidKernel<T>*)k.get();
        gamma_ = sk->getGamma();
        coef0_ = sk->getTerm();
    }   // end else if
}   // end setKernelParams



template <typename T>
boost::shared_ptr<KernelFunc<T> > SVMParams::makeKernel() const
{
    KernelFunc<T> *kernel = NULL;
    if ( isLinear())
        kernel = new LinearKernel<T>();
    else if ( isPoly())
        kernel = new PolyKernel<T>( gamma(), coef0(), degree());
    else if ( isRBF())
        kernel = new GaussianKernel<T>( gamma());
    else if ( isSigmoid())
        kernel = new SigmoidKernel<T>( gamma(), coef0());
    else
    {
        std::cerr << "ERROR: Invalid kernel type of " << kernel_ << " in SVMParams::makeKernel()!" << std::endl;
        assert(false);
    }   // end else

    return boost::shared_ptr<KernelFunc<T> >( kernel);
}   // end makeKernel
