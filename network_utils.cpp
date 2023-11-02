#pragma once
#include "Tensor.h"


// ------ TENSORS LAYERS FUNCTIONS -------


// Apply the sigmoid function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_sigmoid()
{
    PACKAGE_TYPE* iter = this->_begin;

    const PACKAGE_TYPE set0 = _SETZERO();
    const PACKAGE_TYPE set1 = _SET1(1.f);

    const PACKAGE_TYPE value = _SETZERO();

    while (iter < this->_end)
    {
        const PACKAGE_TYPE expValues = _EXP(_SUB(set0, *iter));  // Compute exponent
        const PACKAGE_TYPE sigmoid = _RCP(_ADD(set1, expValues)); // Compute sigmoid

        *iter = sigmoid;

        ++iter;
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE expValues = _EXP(_SUB(set0, *iter));  // Compute exponent
        const PACKAGE_TYPE sigmoid = _RCP(_ADD(set1, expValues)); // Compute sigmoid

        *iter = applyMask<_offset>(sigmoid); // Apply mask
    }
}


// Apply the ReLu function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_ReLu()
{
    PACKAGE_TYPE* iter = this->_begin;

    const PACKAGE_TYPE zeros = _SETZERO();

    while (iter < this->_end)
    {
        *iter = _MAX(*iter, zeros);

        ++iter;
    }

    if constexpr (_offset)
        _MASKSTORE((float*)iter, remainderMask<_offset>(), _MAX(*iter, zeros));
}


// Apply the ReLu derivative function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_ReLu_derivative()
{
    PACKAGE_TYPE* iter = this->_begin;

    const PACKAGE_TYPE zeros = _SETZERO();

    while (iter < this->_end)
    {
        *iter = _AND(_CMP(*iter, zeros, _CMP_GT_OQ), *iter);

        ++iter;
    }

    if constexpr (_offset)
        _MASKSTORE((float*)iter, remainderMask<_offset>(), _AND(_CMP(*iter, zeros, _CMP_GT_OQ), *iter));
}


// Apply a normal distribution on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_normalization()
{
    const float mean = this->mean();

    // Maybe there is a better way to compute sigma reciprocal
    const float sigma = std::sqrt(this->variance(mean));
    const float sigmaReciprocal = 1.f / sigma;

    PACKAGE_TYPE packedMean = _SET1(mean);
    PACKAGE_TYPE packedSigmaReciprocal = _SET1(sigmaReciprocal);

    PACKAGE_TYPE* iter = _begin;

    while (iter < _end)
    {

        *iter = _FMSUB(*iter, packedSigmaReciprocal, packedMean); // Normalize

        ++iter;
    }

    if constexpr (_offset)
    {
        *iter = _FMSUB(*iter, packedSigmaReciprocal, packedMean); // Normalize

        *iter = applyMask<_offset>(*iter); // Apply mask
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

    PACKAGE_TYPE* iter = _begin;

    while (iter < _end)
    {

        *iter = _FMSUB(*iter, packedSigmaReciprocal, packedMean); // Normalize
        *iter = _FMADD(*iter, packedScale, packedShift); // Scale and shift

        ++iter;
    }

    if constexpr (_offset)
    {
        *iter = _FMSUB(*iter, packedSigmaReciprocal, packedMean); // Normalize
        *iter = _FMADD(*iter, packedScale, packedShift); // Scale and shift

        *iter = applyMask<_offset>(*iter); // Apply mask
    }
}