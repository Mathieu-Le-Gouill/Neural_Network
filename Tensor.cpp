#pragma once
#include "Tensor.h"
#include <chrono>
#include <random>
#include "debug.h"


// ------- TENSORS CONSTRUCTORS -------

// Default constructor
template<::std::size_t ...Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor()
{
}


// Copy constructor
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...>::Tensor(const Tensor<Dimensions...>& other)
{
    *this = other;
}


// Rvalue assignement constructor
template <std::size_t... Dimensions>
constexpr Tensor<Dimensions...>::Tensor(Tensor<Dimensions...>&& other) noexcept 
{
    this->_values = other._values;

    other._values = nullptr;
}


// Rvalue assignement constructor for different dimensions Tensor
template <std::size_t... Dimensions>
template <std::size_t... OtherDimensions>
constexpr Tensor<Dimensions...>::Tensor(Tensor<OtherDimensions...>&& other) noexcept 
{
    static_assert((1 * ... * Dimensions) == (1 * ... * OtherDimensions), "Error in Tensor move constructor: incorrect dimensions");
    this->_values = other._values;

    other._values = nullptr;
}


// Fill Tensors with a given value
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...>::Tensor(float value)
{
    const PACKAGE_TYPE packedValue = _SET1(value);

    for(size_t i = 0; i < _numPackages; ++i)
    {
	_values[i] = packedValues;
    }

    if constexpr (_offset)
        _MASKSTORE(_values + _numPackages, remainderMask<_offset>(), packedValue);
}


// Fill the tensor from an initializer list
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...>::Tensor(std::initializer_list<float> values)
{
    debug_assert(values.size() == _size && "Error in Tensor constructor : the given initalizer list does not have a valid size.");

    std::memcpy(_values, values.begin(), _size * sizeof(float));
}


// Fill Tensors with zeros
template <::std::size_t... Dimensions>
Tensor<Dimensions...> zeros()
{
    Tensor<Dimensions...> output;

    const PACKAGE_TYPE packedValues = _SETZERO();

    for(size_t i = 0; i < output._numPackages; ++i)
    {
	output._values[i] = packedValues;
    }

    if constexpr (output._offset)
        _MASKSTORE(output._values + output._numPackages, remainderMask<output._offset>(), packedValues);

    return output;
}


// Fill Tensors with ones
template <::std::size_t... Dimensions>
Tensor<Dimensions...> ones()
{
    Tensor<Dimensions...> output;

    const PACKAGE_TYPE packedValues = _CASTSI_PS(_SET1_EPI32(-1));;

    for(size_t i = 0; i < output._numPackages; ++i)
    {
	output._values[i] = packedValues;
    }

    if constexpr (output._offset)
        _MASKSTORE(output._values + output._numPackages, remainderMask<output._offset>(), packedValues);

    return output;
}


// Fill Tensors with random values with a distribution based on mean and sigma
template <::std::size_t... Dimensions>
Tensor<Dimensions...> rand(float mean, float sigma)
{
    Tensor<Dimensions...> output;

    auto seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();// To get differents epochs 
    std::default_random_engine generator(seed);// Create a generator of random numbers

    std::normal_distribution<float> distribution(mean, sigma);

    for(size_t i = 0; i < output._numPackages; ++i)
    {
        PACKAGE_TYPE randomValues = _mm256_set_ps(distribution(generator), distribution(generator), distribution(generator), distribution(generator),
            distribution(generator), distribution(generator), distribution(generator), distribution(generator));

        output._values[i] = randomValues;
    }

    if constexpr (output._offset)
        _MASKSTORE(output._values + output._numPackages, remainderMask<output._offset>(), _mm256_set_ps(distribution(generator), distribution(generator), distribution(generator), distribution(generator),
            distribution(generator), distribution(generator), distribution(generator), distribution(generator)));

    return output;
}


// Destructor
template<::std::size_t ...Dimensions>
Tensor<Dimensions...>::~Tensor()
{
	if (_values)
		_mm_free(_values);
}


// ------ TENSORS OPERATORS -------


// Copy operator with a Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator=(const Tensor<Dimensions...>& other)
{
    std::memcpy(_values, other._values, _size * sizeof(float));
}


// Copy operator with a rvalue Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator=(Tensor<Dimensions...>&& other) noexcept
{
    if (_values)
        _mm_free(_values);

    this->_values = other._values;

    other._values = nullptr;
}


// Element-wise addition with a Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(const Tensor<Dimensions...>& tensor)
{
    Tensor<Dimensions...> output;

    const PACKAGE_TYPE* iterA = this->_values;
    const PACKAGE_TYPE* iterB = tensor._values;

    PACKAGE_TYPE* iterO = output._values;

    while (iterO < output._end)
    {
        *iterO = _ADD(*iterA, *iterB);

        ++iterA;
        ++iterB;
        ++iterO;
    }

    if constexpr (_offset)
        _MASKSTORE((float*)iterO, remainderMask<_offset>(), _ADD(*iterA, *iterB));

    return output;
}


// Element-wise substraction with a Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(const Tensor<Dimensions...>& tensor)
{
    Tensor<Dimensions...> output;

    const PACKAGE_TYPE* iterA = this->_values;
    const PACKAGE_TYPE* iterB = tensor._values;

    PACKAGE_TYPE* iterO = output._values;

    while (iterO < output._end)
    {
        *iterO = _SUB(*iterA, *iterB);

        ++iterA;
        ++iterB;
        ++iterO;
    }

    if constexpr (_offset)
        _MASKSTORE((float*)iterO, remainderMask<_offset>(), _SUB(*iterA, *iterB));

    return output;
}


// Element-wise multiplication with a Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(const Tensor<Dimensions...>& tensor)
{
    Tensor<Dimensions...> output;

    const PACKAGE_TYPE* iterA = this->_values;
    const PACKAGE_TYPE* iterB = tensor._values;

    PACKAGE_TYPE* iterO = output._values;

    while (iterO < output._end)
    {
        *iterO = _MUL(*iterA, *iterB);

        ++iterA;
        ++iterB;
        ++iterO;
    }

    if constexpr (_offset)
        _MASKSTORE((float*)iterO, remainderMask<_offset>(), _MUL(*iterA, *iterB));

    return output;
}


// Element-wise division with a Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(const Tensor<Dimensions...>& tensor)
{
    Tensor<Dimensions...> output;

    const PACKAGE_TYPE* iterA = this->_values;
    const PACKAGE_TYPE* iterB = tensor._values;

    PACKAGE_TYPE* iterO = output._values;

    while (iterO < output._end)
    {
        *iterO = _DIV(*iterA, *iterB);

        ++iterA;
        ++iterB;
        ++iterO;
    }

    if constexpr (_offset)
        _MASKSTORE((float*)iterO, remainderMask<_offset>(), _DIV(*iterA, *iterB));

    return output;
}


// Element-wise addition with a rvalue Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(Tensor<Dimensions...>&& tensor)
{
    const PACKAGE_TYPE* iterA = this->_values;
    PACKAGE_TYPE* iterB = tensor._values;

    while (iterB < tensor._end)
    {
        *iterB = _ADD(*iterA, *iterB);

        ++iterA;
        ++iterB;
    }

    if constexpr (_offset)
        _MASKSTORE((float*)iterB, remainderMask<_offset>(), _ADD(*iterA, *iterB));

    return tensor;
}


// Element-wise substraction with a rvalue Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(Tensor<Dimensions...>&& tensor)
{
    const PACKAGE_TYPE* iterA = this->_values;
    PACKAGE_TYPE* iterB = tensor._values;

    while (iterB < tensor._end)
    {
        *iterB = _SUB(*iterA, *iterB);

        ++iterA;
        ++iterB;
    }

    if constexpr (_offset)
        _MASKSTORE((float*)iterB, remainderMask<_offset>(), _SUB(*iterA, *iterB));

    return tensor;
}


// Element-wise multiplication with a rvalue Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(Tensor<Dimensions...>&& tensor)
{
	const PACKAGE_TYPE* iterA = this->_values;
	PACKAGE_TYPE* iterB = tensor._values;

	while (iterB < tensor._end)
	{
		*iterB = _MUL(*iterA, *iterB);

		++iterA;
		++iterB;
	}

	if constexpr (_offset)
		_MASKSTORE((float*)iterB, remainderMask<_offset>(), _MUL(*iterA, *iterB));

	return tensor;
}


// Element-wise division with a rvalue Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(Tensor<Dimensions...>&& tensor)
{
	const PACKAGE_TYPE* iterA = this->_values;
	PACKAGE_TYPE* iterB = tensor._values;

	while (iterB < tensor._end)
	{
		*iterB = _DIV(*iterA, *iterB);

		++iterA;
		++iterB;
	}

	if constexpr (_offset)
		_MASKSTORE((float*)iterB, remainderMask<_offset>(), _DIV(*iterA, *iterB));

	return tensor;
}


// Element-wise addition of the tensor with another tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator+=(const Tensor<Dimensions...>& tensor)
{
    PACKAGE_TYPE* iterA = this->_values;
    const PACKAGE_TYPE* iterB = tensor._values;

    while (iterA < this->_end)
    {
        *iterA = _ADD(*iterA, *iterB);

        ++iterA;
        ++iterB;
    }

    if constexpr (_offset)
        _MASKSTORE((float*)iterA, remainderMask<_offset>(), _ADD(*iterA, *iterB));
}


// Element-wise substraction of the tensor with another tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator-=(const Tensor<Dimensions...>& tensor)
{
	PACKAGE_TYPE* iterA = this->_values;
	const PACKAGE_TYPE* iterB = tensor._values;

	while (iterA < this->_end)
	{
		*iterA = _SUB(*iterA, *iterB);

		++iterA;
		++iterB;
	}

	if constexpr (_offset)
		_MASKSTORE((float*)iterA, remainderMask<_offset>(), _SUB(*iterA, *iterB));
}


// Element-wise multiplication of the tensor with another tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator*=(const Tensor<Dimensions...>& tensor)
{
	PACKAGE_TYPE* iterA = this->_values;
	const PACKAGE_TYPE* iterB = tensor._values;

	while (iterA < this->_end)
	{
		*iterA = _MUL(*iterA, *iterB);

		++iterA;
		++iterB;
	}

	if constexpr (_offset)
		_MASKSTORE((float*)iterA, remainderMask<_offset>(), _MUL(*iterA, *iterB));
}


// Element-wise division of the tensor with another tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator/=(const Tensor<Dimensions...>& tensor)
{
	PACKAGE_TYPE* iterA = this->_values;
	const PACKAGE_TYPE* iterB = tensor._values;

	while (iterA < this->_end)
	{
		*iterA = _DIV(*iterA, *iterB);

		++iterA;
		++iterB;
	}

	if constexpr (_offset)
		_MASKSTORE((float*)iterA, remainderMask<_offset>(), _DIV(*iterA, *iterB));
}


// ------ TENSORS BASIC FUNCTIONS -------


// To print the tensor elements
template<::std::size_t ...Dimensions>
void print(const Tensor<Dimensions...>& tensor)
{
    float* iter = (float*)tensor._values;
    ::std::size_t dimensions[] = { Dimensions... };

    for (::std::size_t d = 0; d < tensor._size; ++d) {
        std::cout << *iter << " ";
        ++iter;

        if ((d + 1) % dimensions[0] == 0) {
            std::cout << std::endl;
        }
    }
}


// Reshape the dimensions of the tensor to a compatible one
template<::std::size_t ...Dimensions>
template<::std::size_t ...newDimensions>
constexpr Tensor<newDimensions...> Tensor<Dimensions...>::reshape()
{
    static_assert(_size == (1 * ... * newDimensions), "Error in Tensor reshape : incorrect dimensions");

    Tensor<newDimensions...> output(std::move(*this));
    return output;
}


// Flatten the tensor to a one dimension tensor
template<::std::size_t ...Dimensions>
constexpr Tensor<(1 * ... * Dimensions)> Tensor<Dimensions...>::flatten()
{
    Tensor<_size> output(std::move(*this));
    return output;
}


// Compute the sum of each values in the tensor
template<::std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::sum()
{
    PACKAGE_TYPE packedSum = _SETZERO();

    PACKAGE_TYPE* iter = _values;

    while (iter < _end)
    {
        packedSum = _ADD(*iter, packedSum);

        ++iter;
    }

    if constexpr (_offset)
        packedSum = _ADD(applyMask<_offset>(*iter), packedSum);

    const float sum = horizontal_sum8(packedSum);

    return sum;

}


// Compute the mean of the values in the tensor
template<::std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::mean()
{
    PACKAGE_TYPE* iter = _values;

    PACKAGE_TYPE sum = _SETZERO();

    while (iter < _end)
    {
        sum = _ADD(*iter, sum);

        ++iter;
    }

    if constexpr (_offset)
        sum = _ADD(_MASKLOAD((float*)iter, remainderMask<_offset>()), sum);


    const float mean = horizontal_sum8(sum) / _size;

    return mean;
}


// Compute the variance of the values in the tensor based on a given mean
template<::std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::variance(float mean)
{
    PACKAGE_TYPE deviation = _SETZERO();

    PACKAGE_TYPE* iter = _values;

    while (iter < _end)
    {
        deviation = _ADD(_SUB(*iter, _SET1(mean)), deviation);

        ++iter;
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE maskedDeviation = applyMask<_offset>(_SUB(*iter, _SET1(mean)));

        deviation = _ADD(maskedDeviation, deviation);
    }

    const float variance = horizontal_sum8(_MUL(deviation, deviation)) / _size;

    return variance;
}


// ------ TENSORS MATRIX OPERATIONS -------


// Element-wise addition between tensorC and the element-wise multiplication of tensorA and tensorB
// @return tensorA * tensorB + tensorC
template<::std::size_t ...Dimensions>
Tensor<Dimensions...> multiply_and_add(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC)
{
    Tensor<Dimensions...> output;

    const PACKAGE_TYPE* iterA = tensorA._values;
    const PACKAGE_TYPE* iterB = tensorB._values;
    const PACKAGE_TYPE* iterC = tensorC._values;

    PACKAGE_TYPE* iterO = output._values;

    while (iterO < output._end)
    {
        *iterO = _FMADD(*iterA, *iterB, *iterC);

        ++iterA;
        ++iterB;
        ++iterC;
        ++iterO;
    }

    if constexpr (tensorA._offset)
        _MASKSTORE((float*)iterO, remainderMask<tensorA._offset>(), _FMADD(*iterA, *iterB, *iterC));

    return output;
}


// Element-wise substraction between tensorC and the element-wise multiplication of tensorA and tensorB
// @return tensorA * tensorB - tensorC
template<::std::size_t ...Dimensions>
Tensor<Dimensions...> multiply_and_sub(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC)
{
    Tensor<Dimensions...> output;

    const PACKAGE_TYPE* iterA = tensorA._values;
    const PACKAGE_TYPE* iterB = tensorB._values;
    const PACKAGE_TYPE* iterC = tensorC._values;

    PACKAGE_TYPE* iterO = output._values;

    while (iterO < output._end)
    {
        *iterO = _FMSUB(*iterA, *iterB, *iterC);

        ++iterA;
        ++iterB;
        ++iterC;
        ++iterO;
    }

    if constexpr (tensorA._offset)
        _MASKSTORE((float*)iterO, remainderMask<tensorA._offset>(), _FMSUB(*iterA, *iterB, *iterC));

    return output;
}


// Transpose the tensor
template <::std::size_t cols, ::std::size_t rows, ::std::size_t... rest>
Tensor<cols, rows, rest...> transpose(const Tensor<rows, cols, rest...>& tensor)
{
    Tensor<cols, rows, rest...> output;

    const float* iterA = (float*)tensor._values;
    float* iterB = (float*)output._values;

    for (::std::size_t r = 0; r < rows; ++r)
    {
        for (::std::size_t c = 0; c < cols; ++c)
        {
            iterB[r * cols + c] = iterA[c * rows + r];
        }
    }

    /*const uint16_t& colsA = matrix.m_fullcols;
    const uint16_t& colsB = output.m_fullcols;

    auto* iterA = tensor._values;
    auto* iterB = output._values;

    #pragma omp parallel while schedule(dynamic)
    do
    {
        PACKAGE_TYPE row[PACKAGE_LENGTH];

        row[0] = *iterA;
        iterA += colsA;

        int i;

        for (i = 1; i < PACKAGE_LENGTH; ++i)
        {
            row[i] = (iterA < matrix.m_end) ? *iterA : _SETZERO();
            iterA += colsA;
        }

        #ifdef PACKAGE_M256
        transpose8(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]);

        #elif defined PACKAGE_M128
        _MM_TRANSPOSE4_PS(row[0], row[1], row[2], row[3]);
        #endif

        *iterB = row[0];
        iterB += colsB;

        for (i = 1; i < PACKAGE_LENGTH && iterB < output.m_end; ++i)
        {
            *iterB = row[i];
            iterB += colsB;
        }

        if ((iterB - output.m_values + 1) % colsB == 0)
        {
            iterB += 1 - colsB;
            iterA += (output.m_offcols - PACKAGE_LENGTH - matrix.m_rows) * colsA + 1;
        }
        else
        {
            iterB = posB + 1;
        }

    } while (iterB < output._end);*/

    return output;
}


// ------ TENSORS MATRIX MULTIPLICATION -------


// Matrix multiplication between tensorA and tensorB
template<::std::size_t colsA, ::std::size_t rowsA, ::std::size_t colsB, ::std::size_t... rest>
Tensor<colsB, rowsA, rest...> mul(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsB, colsA, rest...>& tensorB)
{
    constexpr uint16_t offcols = colsB % PACKAGE_LENGTH;
    constexpr ::std::size_t packagePerColumn = colsB / PACKAGE_LENGTH;

    const auto mask = remainderMask<offcols>();

    Tensor<colsB, rowsA, rest...> output;

    const float* iterA = (float*)tensorA._values;
    const float* iterB;

    float* iterO = (float*)output._values;

    #pragma omp parallel do schedule(dynamic)
    do
    {
        ::std::size_t c = 0;

        for (; c < packagePerColumn; c += PACKAGE_LENGTH)
        {
            PACKAGE_TYPE sum = _SETZERO();
            iterB = (float*)tensorB._values + c;

            for (::std::size_t i = 0; i < colsA; ++i)
            {
                const PACKAGE_TYPE valueA = _LOAD1(iterA + i);
                const PACKAGE_TYPE valueB = _LOAD(iterB);

                sum = _FMADD(valueA, valueB, sum);

                iterB += colsB;
            }

            _STORE(iterO + c, sum);
        }

        if constexpr (offcols)
        {
            PACKAGE_TYPE sum = _SETZERO();
            iterB = (float*)tensorB._values + c;

            for (::std::size_t i = 0; i < colsA; ++i)
            {
                const PACKAGE_TYPE valueA = _LOAD1(iterA + i);
                const PACKAGE_TYPE valueB = _LOAD(iterB);

                sum = _FMADD(valueA, valueB, sum);

                iterB += colsB;
            }

            _MASKSTORE(iterO + c, mask, sum);
        }

        iterA += colsA;
        iterO += colsB;

    } while (iterO < (float*)output._end + output._offset);

    return output;
}


// Matrix multiplication between tensorA and the transpose of tensorB
template<::std::size_t colsA, ::std::size_t rowsA, ::std::size_t colsB, ::std::size_t... rest>
Tensor<colsB, rowsA, rest...> mul_transposed(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsA, colsB, rest...>& tensorB)
{
    constexpr ::std::size_t packagePerColumn = colsA / PACKAGE_LENGTH;
    constexpr auto offcols = colsA % PACKAGE_LENGTH;

    const auto mask = remainderMask<offcols>();

    Tensor<colsB, rowsA, rest...> output;

    float* iterA;
    float* iterB;

    float* iterO = (float*)output._values;

    #pragma omp parallel for schedule(dynamic)
    for (::std::size_t r = 0; r < rowsA; ++r)
    {
        for (::std::size_t c = 0; c < colsB; ++c)
        {
            PACKAGE_TYPE sum = _SETZERO();

            iterA = (float*)tensorA._values + colsA * r;
            iterB = (float*)tensorB._values + colsA * c;

            ::std::size_t i = 0;

            for (; i < packagePerColumn; i += PACKAGE_LENGTH)
            {
                const PACKAGE_TYPE valueA = _LOAD(iterA + i);
                const PACKAGE_TYPE valueB = _LOAD(iterB + i);

                sum = _FMADD(valueA, valueB, sum);
            }

            if constexpr (offcols)
            {
                const PACKAGE_TYPE valueA = _MASKLOAD(iterA + i, mask);
                const PACKAGE_TYPE valueB = _MASKLOAD(iterB + i, mask);

                sum = _FMADD(valueA, valueB, sum);
            }

            *iterO = horizontal_sum8(sum);

            ++iterO;
        }
    }

    return output;
}


// Matrix multiplication between tensorA and the transpose of tensorB as a scalar
template<::std::size_t colsA, ::std::size_t rowsA, ::std::size_t... rest>
Tensor<rowsA> mul_transposed_scalar(const Tensor<colsA, rowsA>& tensorA, const Tensor<colsA>& tensorB)
{
    constexpr ::std::size_t packagePerColumn = colsA / PACKAGE_LENGTH;
    constexpr auto offcols = colsA % PACKAGE_LENGTH;

    const auto mask = remainderMask<offcols>();

    Tensor<rowsA> output;

    float* iterA;
    float* iterB;

    float* iterO = (float*)output._values;

    #pragma omp parallel for schedule(dynamic)
    for (::std::size_t r = 0; r < rowsA; ++r)
    {
        PACKAGE_TYPE sum = _SETZERO();

        iterA = (float*)tensorA._values + colsA * r;
        iterB = (float*)tensorB._values;

        ::std::size_t i = 0;

        for (; i < packagePerColumn; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE valueA = _LOAD(iterA + i);
            const PACKAGE_TYPE valueB = _LOAD(iterB + i);

            sum = _FMADD(valueA, valueB, sum);
        }

        if constexpr (offcols)
        {
            const PACKAGE_TYPE valueA = _MASKLOAD(iterA + i, mask);
            const PACKAGE_TYPE valueB = _MASKLOAD(iterB + i, mask);

            sum = _FMADD(valueA, valueB, sum);
        }

        *iterO = horizontal_sum8(sum);

        ++iterO;
    }

    return output;
}


// Matrix multiplication between the transpose of tensorA and tensorB both as a scalar
template<::std::size_t colsA, ::std::size_t colsB, ::std::size_t... rest>
Tensor<colsB, colsA> mul_transposed_scalar(const Tensor<colsA>& tensorA, const Tensor<colsB>& tensorB)
{
    constexpr ::std::size_t packagePerColumn = colsB / PACKAGE_LENGTH;
    constexpr auto offcols = colsB % PACKAGE_LENGTH;

    const auto mask = remainderMask<offcols>();

    Tensor<colsB, colsA> output;

    float* iterO;

    #pragma omp parallel for schedule(dynamic)
    for (::std::size_t c = 0; c < colsA; ++c)
    {
        iterO = (float*)output._values + c * colsB;

        PACKAGE_TYPE valueA = _LOAD1((float*)tensorA._values + c);

        ::std::size_t i = 0;

        for (; i < packagePerColumn; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE valueB = _LOAD((float*)tensorB._values + i);

            const PACKAGE_TYPE result = _MUL(valueA, valueB);

            _STORE(iterO + i, result);
        }

        if constexpr (offcols)
        {
            const PACKAGE_TYPE valueB = _LOAD((float*)tensorB._values + i);

            const PACKAGE_TYPE result = _MUL(valueA, valueB);

            _MASKSTORE(iterO + i, mask, result);
        }
    }

    return output;
}


// ------ TENSORS MASK -------


// Apply to a package value a mask based on a given offset
template<uint16_t offset>
inline static constexpr __m256 applyMask(const __m256& value)
{
    return _CASTSI_PS(_ANDSI(_CASTPS_SI(value), remainderMask<offset>())); // Apply mask
}


#ifdef __AVX2__

// Return a mask based on a given offset
// Only the offset values in the returned mask are set to 1, the others are set to 0
template<uint16_t offset>
inline static constexpr __m256i remainderMask()
{
    static_assert(offset <= PACKAGE_LENGTH, "Error in remainderMask : offset is too big");
    // Make a mask of 8 bytes
    // No need to clip for missingLanes <= 8 because the shift is already good, results in zero
    uint64_t mask = ~(uint64_t)0;
    mask >>= (PACKAGE_LENGTH - offset) * 8;
    // Sign extend these bytes into int32 lanes in AVX vector
    __m128i tmp = _mm_cvtsi64_si128((int64_t)mask);
    return _mm256_cvtepi8_epi32(tmp);
}

#else

// Aligned by 64 bytes
// The load will only touch a single cache line, no penalty for unaligned load
static const int alignas(64) s_remainderLoadMask[16] = {
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };
template<uint16_t offset>
inline __m256i remainderMask()
{
    static_assert(offset <= PACKAGE_LENGTH, "Error in remainderMask : offset is too big");

    // Unaligned load from a constant array
    const int* rsi = &s_remainderLoadMask[offset];
    return _mm256_loadu_si256((const __m256i*)rsi);
}

#endif
