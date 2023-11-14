#pragma once
#include "Tensor.h"


// ------ TENSORS LAYERS FUNCTIONS -------


// Apply the sigmoid function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_sigmoid()
{
    const PACKAGE_TYPE set0 = _SETZERO();
    const PACKAGE_TYPE set1 = _SET1(1.f);

    for (size_t i = 0; i < _numPackages; ++i)
    {
        const PACKAGE_TYPE expValues = _EXP( _SUB(set0, _values[i]));  // Compute exponent
		const PACKAGE_TYPE sigmoid = _RCP( _ADD(set1, expValues)); // Compute sigmoid

        _values[i] = sigmoid;
    }

    if constexpr (_offset)
    {
		const PACKAGE_TYPE expValues = _EXP(_SUB(set0, _values[_numPackages]));  // Compute exponent
		const PACKAGE_TYPE sigmoid = _RCP(_ADD(set1, expValues)); // Compute sigmoid

        _values[_numPackages] = _AND(sigmoid, remainderMask<_offset>()); // Apply mask  
	}
}


// Apply the ReLu function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_ReLu()
{
    const PACKAGE_TYPE zeros = _SETZERO();

    for (size_t i = 0; i < _numPackages; ++i)
    {
		_values[i] = _MAX(_values[i], zeros);
	}

    if constexpr (_offset)
        _values[_numPackages] = _AND( _MAX(_values[_numPackages], zeros), remainderMask<_offset>());
}


// Apply the ReLu derivative function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_ReLu_derivative()
{
    const PACKAGE_TYPE zeros = _SETZERO();

    for (size_t i = 0; i < _numPackages; ++i)
    {
		_values[i] = _AND( _CMP(_values[i], zeros, _CMP_GT_OQ), _values[i]);
	}

    if constexpr (_offset)
		_values[_numPackages] = _AND( _AND( _CMP(_values[_numPackages], zeros, _CMP_GT_OQ), _values[_numPackages]), remainderMask<_offset>());
}


// Apply a normal distribution on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_normalization()
{
    const float mean = this->mean();

    // Maybe there is a better way to compute standard deviation reciprocal
    const float std = std::sqrt(this->variance(mean));
    const float sigmaReciprocal = 1.f / std;

    PACKAGE_TYPE packedMean = _SET1(mean);
    PACKAGE_TYPE packedSigmaReciprocal = _SET1(sigmaReciprocal);

    for (size_t i = 0; i < _numPackages; ++i)
    {
		_values[i] = _FMSUB(_values[i], packedSigmaReciprocal, packedMean); // Normalize
	}

    if constexpr (_offset)
    {
        const PACKAGE_TYPE value = _FMSUB(_values[_numPackages], packedSigmaReciprocal, packedMean);

        _values[_numPackages] = _AND(value, remainderMask<_offset>());
    }
}


// Normalize the tensor based on a given mean and variance then shift and scale the values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::norm_shift_and_scale(float mean, float variance, float shift, float scale)
{
    debug_assert(variance != 0 && "Error in Tensor apply_shift_and_scale : the given variance is equal to 0.");

    // Maybe there is a better way to compute sigma reciprocal
    const float sigmaReciprocal = 1.f / std::sqrtf(variance);

    PACKAGE_TYPE packedShift = _SET1(shift);
    PACKAGE_TYPE packedScale = _SET1(scale);

    PACKAGE_TYPE packedMean = _SET1(mean);
    PACKAGE_TYPE packedSigmaReciprocal = _SET1(sigmaReciprocal);

    for (size_t i = 0; i < _numPackages; ++i)
    {
        _values[i] = _FMSUB(_values[i], packedSigmaReciprocal, packedMean); // Normalize
        _values[i] = _FMADD(_values[i], packedScale, packedShift); // Scale and shift
    }

    if constexpr (_offset)
    {
        _values[_numPackages] = _FMSUB(_values[_numPackages], packedSigmaReciprocal, packedMean); // Normalize
        _values[_numPackages] = _FMADD(_values[_numPackages], packedScale, packedShift); // Scale and shift

        _values[_numPackages] = _AND(_values[_numPackages], remainderMask<_offset>()); // Apply mask
    }
}