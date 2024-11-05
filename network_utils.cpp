#pragma once
#include "Tensor.h"


// ------ TENSORS LAYERS FUNCTIONS -------

inline float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Apply the sigmoid function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_sigmoid()
{
    for(int i=0; i<_size; i++)
    {
        _values[i] = sigmoid(_values[i]);// Compute the matrix sigmoid
    }


    /*
    const PACKAGE_TYPE set0 = _SETZERO();
    const PACKAGE_TYPE set1 = _SET1(1.f);

    size_t i = 0;

    #pragma omp parallel for schedule(dynamic)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

        const PACKAGE_TYPE expValues = _EXP( _SUB(set0, packedValues));  // Compute exponent
		const PACKAGE_TYPE sigmoid = _RCP( _ADD(set1, expValues)); // Compute sigmoid

        _STORE(_values + i, sigmoid); // Store values in memory
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

		const PACKAGE_TYPE expValues = _EXP( _SUB(set0, packedValues));  // Compute exponent
		const PACKAGE_TYPE sigmoid = _RCP( _ADD(set1, expValues)); // Compute sigmoid

        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), sigmoid); // Store values in memory
	}*/
}


// Apply the ReLu function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_ReLU()
{
    const PACKAGE_TYPE zeros = _SETZERO();

    size_t i = 0;

    #pragma omp parallel for schedule(dynamic)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

        _STORE(_values + i, _MAX(packedValues, zeros)); // Store values in memory
	}

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), _MAX(packedValues, zeros)); // Store values in memory
    }
}


// Apply the ReLu derivative function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_ReLU_derivative()
{
    const PACKAGE_TYPE zeros = _SETZERO();

    size_t i = 0;

    #pragma omp parallel for schedule(dynamic)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

        const PACKAGE_TYPE mask = _CMP(packedValues, zeros, _CMP_GT_OQ); // Create mask

        _STORE(_values + i, _AND(mask, packedValues)); // Store values in memory
	}

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

		const PACKAGE_TYPE mask = _CMP(packedValues, zeros, _CMP_GT_OQ); // Create mask

		_MASKSTORE(_values + i, remainderMaskSI<_offset>(), _AND(mask, packedValues)); // Store values in memory
    }
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

    size_t i = 0;

    #pragma omp parallel for schedule(dynamic)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

        _STORE(_values + i, _FMSUB(packedValues, packedSigmaReciprocal, packedMean)); // Normalize
	}

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

        const PACKAGE_TYPE result = _FMSUB(packedValues, packedSigmaReciprocal, packedMean);

        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), result); // Store values in memory
    }
}


// Normalize the tensor based on a given mean and variance then shift and scale the values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::norm_shift_and_scale(float mean, float variance, float shift, float scale)
{
    debug_assert(variance != 0 && "Error in Tensor apply_shift_and_scale : the given variance is null.");

    // Maybe there is a better way to compute sigma reciprocal
    const float sigmaReciprocal = 1.f / std::sqrtf(variance);

    PACKAGE_TYPE packedShift = _SET1(shift);
    PACKAGE_TYPE packedScale = _SET1(scale);

    PACKAGE_TYPE packedMean = _SET1(mean);
    PACKAGE_TYPE packedSigmaReciprocal = _SET1(sigmaReciprocal);

    size_t i = 0;

    #pragma omp parallel for schedule(dynamic)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

        const PACKAGE_TYPE norm = _FMSUB(packedValues, packedSigmaReciprocal, packedMean); // Normalize
        const PACKAGE_TYPE result = _FMADD(norm, packedScale, packedShift); // Scale and shift

        _STORE(_values + i, result); // Store values in memory
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

        const PACKAGE_TYPE norm = _FMSUB(packedValues, packedSigmaReciprocal, packedMean); // Normalize
        const PACKAGE_TYPE result = _FMADD(norm, packedScale, packedShift); // Scale and shift

        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), result); // Store values in memory
    }
}


// Apply dropout on the tensor values setting random values to zero based on a given rate
// Return the build mask to be able to apply the same dropout on the backward pass
template<::std::size_t ...Dimensions>
Tensor<Dimensions...> Tensor<Dimensions...>::apply_dropout(float rate)
{
    debug_assert(rate > 0.f && rate < 1.f && "Error in Tensor apply_dropout : the given rate is not between 0 and 1.");

    Tensor<Dimensions...> dropoutMask;
    dropoutMask.init();

	const PACKAGE_TYPE packedRate = _SET1(rate);

    auto seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();// To get differents epochs 
    std::default_random_engine generator(seed);// Create a generator of random numbers

    std::uniform_real_distribution<float> distribution(0, 1);

    size_t i = 0;

    #pragma omp parallel for schedule(dynamic)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        // Generate random values between 0 and 1
        const PACKAGE_TYPE randomValues = _mm256_set_ps(distribution(generator), distribution(generator),
                                                        distribution(generator), distribution(generator),
                                                        distribution(generator), distribution(generator),
                                                        distribution(generator), distribution(generator));

        const PACKAGE_TYPE mask = _CMP(randomValues, packedRate, _CMP_GT_OQ); // Create mask

        const PACKAGE_TYPE packedValues = _LOAD(this->_values + i); // Load values from memory

        _STORE(this->_values + i, _AND(packedValues, mask)); // Store values in memory

        _STORE(dropoutMask._values + i, mask); // Save mask
	}

    if constexpr (_offset)
    {
        // Generate random values between 0 and 1
        const PACKAGE_TYPE randomValues = _mm256_set_ps(distribution(generator), distribution(generator),
                                                        distribution(generator), distribution(generator),
                                                        distribution(generator), distribution(generator),
                                                        distribution(generator), distribution(generator));

		const PACKAGE_TYPE mask = _CMP(randomValues, packedRate, _CMP_GT_OQ); // Create mask

        const PACKAGE_TYPE packedValues = _LOAD(this->_values + i); // Load values from memory

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _AND(packedValues, mask)); // Store values in memory

        _MASKSTORE(dropoutMask._values + i, remainderMaskSI<_offset>(), mask); // Save mask
	}

    return dropoutMask;
}


// Apply the given mask on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_mask(const Tensor<Dimensions...>& mask)
{
    size_t i = 0;

    #pragma omp parallel for schedule(dynamic)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory
		const PACKAGE_TYPE packedMask = _LOAD(mask._values + i); // Load mask from memory

		_STORE(_values + i, _AND(packedValues, packedMask)); // Apply mask
	}

    if constexpr (_offset)
    {
		const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory
        const PACKAGE_TYPE packedMask = _LOAD(mask._values + i); // Load mask from memory

        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), _AND(packedValues, packedMask)); // Apply mask
	}
}

// Apply the softmax function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_Softmax()
{
    const float maxValue = this->max(); // max values of the tensor

    // Compute sum
    const PACKAGE_TYPE packedMax = _SET1(maxValue);

    PACKAGE_TYPE packedSum;

    if constexpr (_size >= PACKAGE_LENGTH)
    {
        packedSum = _EXP( _SUB( _LOAD(_values), packedMax));  // Compute exponent

        size_t i = PACKAGE_LENGTH;

        #pragma omp parallel for schedule(dynamic)
        for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

            const PACKAGE_TYPE expValues = _EXP( _SUB(packedValues, packedMax));  // Compute exponent

			packedSum = _ADD(expValues, packedSum);
        }

        if constexpr (_offset)
        {
            const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

            const PACKAGE_TYPE expValues = _EXP( _SUB(packedValues, packedMax));  // Compute exponent

            const PACKAGE_TYPE maskedValues = _AND(expValues, remainderMask<_offset>()); // Apply mask

            packedSum = _ADD(maskedValues, packedSum);
        }
    }
    else
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values); // Load values from memory

        const PACKAGE_TYPE expValues = _EXP( _SUB(packedValues, packedMax));  // Compute exponent

        packedSum = _AND(expValues, remainderMask<_offset>());
    }

    const float totalSum = horizontal_sum8(packedSum); // Sum all values

    PACKAGE_TYPE packedRCPSum = _SET1(1.f/totalSum);

    size_t i = 0;

    #pragma omp parallel for schedule(dynamic)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

		const PACKAGE_TYPE expValues = _EXP( _SUB(packedValues, packedMax));  // Compute exponent
		const PACKAGE_TYPE softmax = _MUL(expValues, packedRCPSum); // Compute softmax

        _STORE(_values + i, softmax); // Store values in memory
	}

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values + i); // Load values from memory

        const PACKAGE_TYPE expValues = _EXP(_SUB(packedValues, packedMax));  // Compute exponent
        const PACKAGE_TYPE softmax = _MUL(expValues, packedRCPSum); // Compute softmax

        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), softmax); // Store values in memory
	}
}


// Apply the softmax derivative function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_Softmax_derivative()
{
    float upstreamLossGradient[_size];

    std::memcpy(upstreamLossGradient, _values, _size * sizeof(float));

    float* lossGradientValue = (float*)_values;

    #pragma omp parallel for schedule(dynamic)
    // For each loss gradient values
    for (size_t i = 0; i < _size; i++)
    {
        const float& upstreamLossGradientValue = upstreamLossGradient[i];

        // For each upstream loss gradient values
        for (size_t j = 0; j < _size; j++)
        {
            // Compute the loss gradient
            lossGradientValue[i] += (i == j) ?
                upstreamLossGradientValue * (1 - upstreamLossGradientValue) : - upstreamLossGradientValue * (upstreamLossGradient[j]);
        }

    }
}


// Compute the cross entropy loss function
template<::std::size_t ...Dimensions>
float Cross_Entropy_loss(const Tensor<Dimensions...>& prediction, const Tensor<Dimensions...>& labels)
{
    constexpr size_t offset = Tensor<Dimensions...>::_offset;
    constexpr size_t size = Tensor<Dimensions...>::_size;

    PACKAGE_TYPE packedSum;

    if constexpr (size >= PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE log = _LOG( _LOAD(prediction._values)); // Load labels from memory

        packedSum = _MUL( _LOAD(labels._values), log);

        size_t i = PACKAGE_LENGTH;

        #pragma omp parallel for schedule(dynamic)
        for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE labelsValues = _LOAD(labels._values + i); // Load labels from memory
            const PACKAGE_TYPE predictionValues = _LOAD(prediction._values + i); // Load prediction from memory

            packedSum = _FMADD(labelsValues, _LOG(predictionValues), packedSum); // Compute loss
        }

        if constexpr (offset)
        {
            const PACKAGE_TYPE labelsValues = _MASKLOAD(labels._values + i, remainderMaskSI<offset>()); // Load labels from memory
            const PACKAGE_TYPE predictionValues = _LOG(prediction._values + i); // Load prediction from memory

            const PACKAGE_TYPE maskedPredictionsLog = _AND( _LOG(predictionValues), remainderMask<offset>()); // Apply mask on the log of the prediction

            packedSum = _FMADD(labelsValues, maskedPredictionsLog, packedSum); // Compute loss
        }
    }
    else 
    {
        const PACKAGE_TYPE labelsValues = _LOAD(labels._values); // Load labels from memory
		const PACKAGE_TYPE predictionValues = _LOAD(prediction._values); // Load prediction from memory

		const PACKAGE_TYPE values = _MUL(labelsValues, _LOG(predictionValues));

		packedSum = _AND(values, remainderMask<offset>()); // Apply mask
	}

	const float ce = horizontal_sum8(packedSum); // Sum all values

	return -ce;
}