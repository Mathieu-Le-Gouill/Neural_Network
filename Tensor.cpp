#pragma once
#include "Tensor.h"
#include <chrono>
#include <random>
#include "debug.h"


// TODO : Look for OpenMP for threading and compare it with manual threading
//        See for the other implementation of the argmax function
//        See for the other implementation of the transpose function
//        Think about how the functions can be unrolled to use all the registers
//        Check for an implementations for ones


// ------- TENSORS ALIAS -------


// Alias for the Tensor of two dimensions
template <::std::size_t cols, ::std::size_t rows>
using Matrix = Tensor<cols, rows>;

// Alias for the Tensor of one dimension
template <::std::size_t size>
using Vector = Tensor<size>;


// ------- TENSORS CONSTRUCTORS -------

// Default constructor
template<::std::size_t ...Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor()
{
    _values = static_cast<PACKAGE_TYPE*>(_mm_malloc(_size * sizeof(float) + PACKAGE_ALIGNEMENT, PACKAGE_ALIGNEMENT));
}


// Copy constructor
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...>::Tensor(const Tensor<Dimensions...>& other)
{
    _values = static_cast<PACKAGE_TYPE*>(_mm_malloc(_size * sizeof(float) + PACKAGE_ALIGNEMENT, PACKAGE_ALIGNEMENT));

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
    _values = static_cast<PACKAGE_TYPE*>(_mm_malloc(_size * sizeof(float) + PACKAGE_ALIGNEMENT, PACKAGE_ALIGNEMENT));

    const PACKAGE_TYPE packedValues = _SET1(value);

    for (size_t i = 0; i < _numPackages; ++i)
    {
        _values[i] = packedValues;
    }

    if constexpr (_offset)
        _values[_numPackages] = _AND(packedValues, remainderMask<_offset>());
        //_MASKSTORE((float*)& _values[_numPackages], remainderMask<_offset>(), packedValues);
}


// Fill the tensor from an initializer list
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...>::Tensor(std::initializer_list<float> values)
{
    debug_assert(values.size() == _size && "Error in Tensor constructor : the given initalizer list does not have a valid size.");

    _values = static_cast<PACKAGE_TYPE*>(_mm_malloc(_size * sizeof(float) + PACKAGE_ALIGNEMENT, PACKAGE_ALIGNEMENT));

    std::memcpy(_values, values.begin(), _size * sizeof(float));
}


// Fill Tensors with zeros
template <::std::size_t... Dimensions>
Tensor<Dimensions...> zeros()
{
    Tensor<Dimensions...> output;

    constexpr size_t numPackages = output._numPackages;
    constexpr uint16_t offset = output._offset;

    const PACKAGE_TYPE packedValues = _SETZERO();

    for (size_t i = 0; i < numPackages; ++i)
    {
        output._values[i] = packedValues;
    }

    if constexpr (offset)
        output._values[numPackages] = _AND(packedValues, remainderMask<offset>());

    return output;
}


// Fill Tensors with ones
template <::std::size_t... Dimensions>
Tensor<Dimensions...> ones()
{
    Tensor<Dimensions...> output;

    constexpr size_t numPackages = output._numPackages;
    constexpr uint16_t offset = output._offset;

    const PACKAGE_TYPE packedValues = _SET1(1.f);

    for (size_t i = 0; i < numPackages; ++i)
    {
        output._values[i] = packedValues;
    }

    if constexpr (offset)
        output._values[numPackages] = _AND(packedValues, remainderMask<offset>());

    return output;
}


// Fill Tensors with random values from a normal distribution based on mean and the standard deviation
template <::std::size_t... Dimensions>
Tensor<Dimensions...> normal(float mean, float std)
{
    Tensor<Dimensions...> output;

    constexpr size_t numPackages = output._numPackages;
    constexpr uint16_t offset = output._offset;

    auto seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();// To get differents epochs 
    std::default_random_engine generator(seed);// Create a generator of random numbers

    std::normal_distribution<float> distribution(mean, std);

    for (size_t i = 0; i < numPackages; ++i)
    {
        PACKAGE_TYPE randomValues = _mm256_set_ps(distribution(generator), distribution(generator), distribution(generator), distribution(generator),
                                                  distribution(generator), distribution(generator), distribution(generator), distribution(generator));

        output._values[i] = randomValues;
    }

    if constexpr (offset)
    {
        const PACKAGE_TYPE randomValues = _mm256_set_ps(distribution(generator), distribution(generator), distribution(generator), distribution(generator),
            											distribution(generator), distribution(generator), distribution(generator), distribution(generator));
        
        output._values[numPackages] = _AND(randomValues, remainderMask<offset>());
    }

    return output;
}


// Destructor
template<::std::size_t ...Dimensions>
Tensor<Dimensions...>::~Tensor()
{
    if(_values)
        _mm_free(_values);  
}


// ------ TENSORS OPERATORS -------


// Copy operator with a Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator=(const Tensor<Dimensions...>& other)
{
    std::memcpy(this->_values, other._values, _size * sizeof(float));
}


// Copy operator with a rvalue Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator=(Tensor<Dimensions...>&& other) noexcept
{
    if(_values)
        _mm_free(_values);
   
    this->_values = other._values;

    other._values = nullptr;
}


// Element-wise addition with a Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(const Tensor<Dimensions...>& tensor)
{
    Tensor<Dimensions...> output;

    for (size_t i = 0; i < _numPackages; ++i)
    {
        output._values[i] = _ADD(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        output._values[_numPackages] = _AND( _ADD(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());

    return output;
}


// Element-wise substraction with a Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(const Tensor<Dimensions...>& tensor)
{
    Tensor<Dimensions...> output;

    for (size_t i = 0; i < _numPackages; ++i)
    {
        output._values[i] = _SUB(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        output._values[_numPackages] = _AND( _SUB(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());

    return output;
}


// Element-wise multiplication with a Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(const Tensor<Dimensions...>& tensor)
{
    Tensor<Dimensions...> output;
    for (size_t i = 0; i < _numPackages; ++i)
    {
        output._values[i] = _MUL(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        output._values[_numPackages] = _AND( _MUL(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());

    return output;
}


// Element-wise division with a Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(const Tensor<Dimensions...>& tensor)
{
    Tensor<Dimensions...> output;

    for (size_t i = 0; i < _numPackages; ++i)
    {
        output._values[i] = _DIV(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        output._values[_numPackages] = _AND( _DIV(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());

    return output;
}


// Element-wise addition with a rvalue Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(Tensor<Dimensions...>&& tensor)
{
    for (size_t i = 0; i < _numPackages; ++i)
    {
        tensor._values[i] = _ADD(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        tensor._values[_numPackages] = _AND( _ADD(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());

    return tensor;
}


// Element-wise substraction with a rvalue Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(Tensor<Dimensions...>&& tensor)
{
    for (size_t i = 0; i < _numPackages; ++i)
    {
        tensor._values[i] = _SUB(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        tensor._values[_numPackages] = _AND( _SUB(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());

    return tensor;
}


// Element-wise multiplication with a rvalue Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(Tensor<Dimensions...>&& tensor)
{
    for (size_t i = 0; i < _numPackages; ++i)
    {
        tensor._values[i] = _MUL(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        tensor._values[_numPackages] = _AND( _MUL(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());

    return tensor;
}


// Element-wise division with a rvalue Tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(Tensor<Dimensions...>&& tensor)
{
    for (size_t i = 0; i < _numPackages; ++i)
    {
        tensor._values[i] = _DIV(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        tensor._values[_numPackages] = _AND( _DIV(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());

    return tensor;
}


// Element-wise addition of the tensor with another tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator+=(const Tensor<Dimensions...>& tensor)
{
    for (size_t i = 0; i < _numPackages; ++i)
    {
        this->_values[i] = _ADD(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        this->_values[_numPackages] = _AND( _ADD(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());
}


// Element-wise substraction of the tensor with another tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator-=(const Tensor<Dimensions...>& tensor)
{
    for (size_t i = 0; i < _numPackages; ++i)
    {
        this->_values[i] = _SUB(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        this->_values[_numPackages] = _AND( _SUB(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());
}


// Element-wise multiplication of the tensor with another tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator*=(const Tensor<Dimensions...>& tensor)
{
    for (size_t i = 0; i < _numPackages; ++i)
    {
        this->_values[i] = _MUL(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        this->_values[_numPackages] = _AND( _MUL(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());
}


// Element-wise division of the tensor with another tensor of the same dimensions
template<::std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator/=(const Tensor<Dimensions...>& tensor)
{
    for (size_t i = 0; i < _numPackages; ++i)
    {
        this->_values[i] = _DIV(this->_values[i], tensor._values[i]);
    }

    if constexpr (_offset)
        this->values[_numPackages] = _AND( _DIV(this->_values[_numPackages], tensor._values[_numPackages]), remainderMask<_offset>());
}


// ------ TENSORS BASIC FUNCTIONS -------


// To print the tensor elements
template<::std::size_t ...Dimensions>
void print(const Tensor<Dimensions...>& tensor)
{
    float* values = (float*)tensor._values;
    ::std::size_t dimensions[] = { Dimensions... };

    for (::std::size_t d = 0; d < tensor._size; ++d) {
        std::cout << *values << " ";
        ++values;

        if ((d + 1) % dimensions[0] == 0) {
            std::cout << std::endl;
        }
    }
}


// Compute the absolute value of each element in the tensor
template<::std::size_t ...Dimensions>
Tensor<Dimensions...> abs(const Tensor<Dimensions...>& tensor)
{
    Tensor<Dimensions...> output;

    constexpr size_t numPackages = tensor._numPackages;
    constexpr uint16_t offset = tensor._offset;

    auto minus1 = _SET1_EPI32(-1);
    const PACKAGE_TYPE absMask = _CASTSI_PS(_mm256_srli_epi32(minus1, 1));

    for (size_t i = 0; i < numPackages; ++i)
    {
        output._values[i] = _AND(absMask, tensor._values[i]); // Clear the sign bit
    }

    if constexpr (offset)
        output._values[numPackages] = _AND( _AND(absMask, tensor._values[numPackages]), remainderMask<offset>());

    return output;
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
    PACKAGE_TYPE packedSum;

    if constexpr (_size >= PACKAGE_LENGTH)
    {
        packedSum = _values[0];

        for (size_t i = 1; i < _numPackages; ++i)
        {
            packedSum = _ADD(_values[i], packedSum);
        }

        if constexpr (_offset)
            packedSum = _ADD( _AND(_values[_numPackages], remainderMask<_offset>()), packedSum);
    }
    else
        packedSum = _AND(_values[0], remainderMask<_size>());

    const float sum = horizontal_sum8(packedSum);

    return sum;
}


// Find the index of the maximum value in the tensor
// ref : https://en.algorithmica.org/hpc/algorithms/argmin/
template<::std::size_t ...Dimensions>
constexpr size_t Tensor<Dimensions...>::argmax()
{
    PACKAGE_TYPE maxValues;

    size_t argmax = 0;
    float max;

    if constexpr (_size > PACKAGE_LENGTH)
    {
        max = horizontal_max8(_values[0]);
        maxValues = _SET1(max);

        for (size_t i = 1; i < _numPackages; ++i)
        {
            maxValues = _MAX(_values[i], maxValues);

            auto compMask = _CMPGT_EPI32( _CASTPS_SI(maxValues), _CASTPS_SI(_values[i]));

            if (!_mm256_testz_si256(compMask, compMask)) [[unlikely]]
            {
                max = horizontal_max8(maxValues);
                argmax = i;

                maxValues = _SET1(max);
            }
				
        }

        if constexpr (_offset)
        {
            float* values = (float*)_values;

            for (size_t i = _size - _offset; i < _size; ++i)
            {
                if (values[i] > max)
                {
                    max = values[i];
                    argmax = i;

                    for (size_t r = i+1; r < _size; ++r)
                    {
                        if (values[r] > max)
                        {
                            max = values[r];
                            argmax = r;
                        }
                    }

                    return argmax;
                }
            }
        }
    }
    else
	{
        float* values = (float*)_values;
        max = values[0];

        for (uint16_t i = 1; i < _offset; ++i)
        {
            if (values[i] > max)
            {
                max = values[i];
                argmax = i;
            }
        }

        return argmax;
	}

    float* values = (float*)_values;

    for (size_t i = argmax; i < argmax + PACKAGE_LENGTH; ++i)
    {
        if (values[i] == max)
            return i;
	}

	return -1;
}


// Find the maximum value in the tensor
template<::std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::max()
{
    PACKAGE_TYPE maxValues;

    if constexpr (_size > PACKAGE_LENGTH)
    {
        maxValues = _values[0];

        for (size_t i = 1; i < _numPackages; ++i)
        {
            maxValues = _MAX(_values[i], maxValues);
        }

        if constexpr (_offset)
        {
            PACKAGE_TYPE minimum = _SET1(-FLT_MAX);
            constexpr int mask = (1 << _offset) - 1;

            maxValues = _MAX(_BLEND(minimum, _values[_numPackages], mask), maxValues);
        }


    }
    else
    {
        PACKAGE_TYPE minimum = _SET1(-FLT_MAX);
        constexpr int mask = (1 << _offset) - 1;

        maxValues = _BLEND(minimum, _values[_numPackages], mask);
    }
       

    float max = horizontal_max8(maxValues);

    return max;
}


// Compute the mean of the values in the tensor
template<::std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::mean()
{
    const float mean = sum() / _size;

    return mean;
}


// Compute the variance of the values in the tensor based on a given mean
template<::std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::variance(float mean)
{
    PACKAGE_TYPE packedMean = _SET1(mean);

    PACKAGE_TYPE deviation;

    if constexpr (_size > PACKAGE_LENGTH)
    {
        deviation = _SUB(_values[0], packedMean);

        for (size_t i = 1; i < _numPackages; ++i)
        {
            deviation = _ADD( _SUB(_values[i], packedMean), deviation);
        }

        if constexpr (_offset)
        {
            const PACKAGE_TYPE maskedValues = _AND(_values[_numPackages], remainderMask<_offset>());

            deviation = _ADD( _SUB(maskedValues, packedMean), deviation);
        }
    }
    else
        deviation = _AND(_SUB(_values[0], packedMean), remainderMask<_size>());

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

    constexpr size_t numPackages = output._numPackages;
    constexpr uint16_t offset = output._offset;

    for (size_t i = 0; i < numPackages; ++i)
    {
        output._values[i] = _FMADD(tensorA._values[i], tensorB._values[i], tensorC._values[i]);
    }

    if constexpr (offset)
    {
        const PACKAGE_TYPE value = _FMADD(tensorA._values[numPackages], tensorB._values[numPackages], tensorC._values[numPackages]);

        output._values[numPackages] = _AND(value, remainderMask<offset>());
    }

    return output;
}


// Element-wise substraction between tensorC and the element-wise multiplication of tensorA and tensorB
// @return tensorA * tensorB - tensorC
template<::std::size_t ...Dimensions>
Tensor<Dimensions...> multiply_and_sub(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC)
{
    Tensor<Dimensions...> output;

    constexpr size_t numPackages = output._numPackages;
    constexpr uint16_t offset = output._offset;

    for (size_t i = 0; i < numPackages; ++i)
    {
        output._values[i] = _FMSUB(tensorA._values[i], tensorB._values[i], tensorC._values[i]);
    }

    if constexpr (offset)
    {
        const PACKAGE_TYPE value = _FMSUB(tensorA._values[numPackages], tensorB._values[numPackages], tensorC._values[numPackages]);

        output._values[numPackages] = _AND(value, remainderMask<offset>());
    }

    return output;
}


// Transpose the tensor
template <::std::size_t cols, ::std::size_t rows, ::std::size_t... rest>
Tensor<cols, rows, rest...> transpose(const Tensor<rows, cols, rest...>& tensor)
{
    Tensor<cols, rows, rest...> output;

    const float* valuesA = (float*)tensor._values;
    float* valuesB = (float*)output._values;

    for (::std::size_t r = 0; r < rows; ++r)
    {
        for (::std::size_t c = 0; c < cols; ++c)
        {
            valuesB[r * cols + c] = valuesA[c * rows + r];
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
    constexpr uint16_t offCols = colsB % PACKAGE_LENGTH;

    const auto mask = remainderMaskSI<offCols>();

    Tensor<colsB, rowsA, rest...> output;

    const float* iterA = (float*)tensorA._values;
    const float* iterB;

    float* iterO = (float*)output._values;

    #pragma omp parallel for schedule(dynamic)
    for (::std::size_t r = 0; r < rowsA; ++r)
    {
        ::std::size_t c = 0;

        for (; c + PACKAGE_LENGTH < colsB; c += PACKAGE_LENGTH)
        {
            iterB = (float*)tensorB._values + c;

            PACKAGE_TYPE sum = _MUL(_LOAD1(iterA), _LOAD(iterB));

            for (::std::size_t i = 1; i < colsA; ++i)
            {
                const PACKAGE_TYPE packageA = _LOAD1(iterA + i);
                const PACKAGE_TYPE packageB = _LOAD(iterB + i * colsB);

                sum = _FMADD(packageA, packageB, sum);
            }

            _STORE(&iterO[c], sum);
        }

        if constexpr (offCols)
        {
            iterB = (float*)tensorB._values + c;

            PACKAGE_TYPE sum = _MUL(_LOAD1(iterA), _LOAD(iterB));

            for (::std::size_t i = 1; i < colsA; ++i)
            {
                const PACKAGE_TYPE packageA = _LOAD1(iterA + i);
                const PACKAGE_TYPE packageB = _LOAD(iterB + i * colsB);

                sum = _FMADD(packageA, packageB, sum);
            }

            _MASKSTORE(&iterO[c], mask, sum);
        }

        iterA += colsA;
        iterO += colsB;
    }

    return output;
}


// Matrix multiplication between tensorA and the transpose of tensorB
template<::std::size_t colsA, ::std::size_t rowsA, ::std::size_t colsB, ::std::size_t... rest>
Tensor<colsB, rowsA, rest...> mul_transposed(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsA, colsB, rest...>& tensorB)
{
    constexpr auto offcols = colsA % PACKAGE_LENGTH;

    const auto mask = remainderMaskSI<offcols>();

    Tensor<colsB, rowsA, rest...> output;

    float* iterA;
    float* iterB;

    float* iterO = (float*)output._values;

#pragma omp parallel for schedule(dynamic)
    for (::std::size_t r = 0; r < rowsA; ++r)
    {
        for (::std::size_t c = 0; c < colsB; ++c)
        {
            iterA = (float*)tensorA._values + colsA * r;
            iterB = (float*)tensorB._values + colsA * c;

            PACKAGE_TYPE sum;

            if constexpr (colsA > PACKAGE_LENGTH)
            {
                PACKAGE_TYPE sum = _MUL(_LOAD(iterA), _LOAD(iterB));

                ::std::size_t i = 1;

                for (; i + PACKAGE_LENGTH < colsA; i += PACKAGE_LENGTH)
                {
                    const PACKAGE_TYPE packageA = _LOAD(iterA + i);
                    const PACKAGE_TYPE packageB = _LOAD(iterB + i);

                    sum = _FMADD(packageA, packageB, sum);
                }

                if constexpr (offcols)
                {
                    const PACKAGE_TYPE packageA = _MASKLOAD(iterA + i, mask);
                    const PACKAGE_TYPE packageB = _MASKLOAD(iterB + i, mask);

                    sum = _FMADD(packageA, packageB, sum);
                }
            }
            else
                sum = _AND( _MUL( _LOAD(iterA), _LOAD(iterB)), remainderMask<colsA>());

            *iterO = horizontal_sum8(sum);

            ++iterO;
        }
    }

    return output;
}


// Matrix multiplication between tensorA and the transpose of tensorB as a scalar
template<::std::size_t colsA, ::std::size_t rowsA>
Tensor<rowsA> mul_transposed_scalar(const Tensor<colsA, rowsA>& tensorA, const Tensor<colsA>& tensorB)
{
    constexpr auto offCols = colsA % PACKAGE_LENGTH;

    const auto mask = remainderMaskSI<offCols>();

    Tensor<rowsA> output;

    float* iterA;
    float* iterB;

    float* iterO = (float*)output._values;

    #pragma omp parallel for schedule(dynamic)
    for (::std::size_t r = 0; r < rowsA; ++r)
    {
        iterA = (float*)tensorA._values + colsA * r;
        iterB = (float*)tensorB._values;

        PACKAGE_TYPE sum;

        if constexpr (colsA > PACKAGE_LENGTH)
        {
			sum = _MUL( _LOAD(iterA), _LOAD(iterB));

			::std::size_t i = 1;

			for (; i + PACKAGE_LENGTH < colsA; i += PACKAGE_LENGTH)
			{
				const PACKAGE_TYPE packageA = _LOAD(iterA + i);
				const PACKAGE_TYPE packageB = _LOAD(iterB + i);

				sum = _FMADD(packageA, packageB, sum);
			}

            if constexpr (offCols)
            {
				const PACKAGE_TYPE packageA = _MASKLOAD(iterA + i, mask);
                const PACKAGE_TYPE packageB = _MASKLOAD(iterB + i, mask);

				sum = _FMADD(packageA, packageB, sum);
			}
		}
		else
            sum = _AND( _MUL( _LOAD(iterA), _LOAD(iterB)), remainderMask<colsA>());

        *iterO = horizontal_sum8(sum);

        ++iterO;
    }

    return output;
}


// Matrix multiplication between the transpose of tensorA and tensorB both as a scalar
template<::std::size_t colsA, ::std::size_t colsB>
Tensor<colsB, colsA> mul_transposed_scalar(const Tensor<colsA>& tensorA, const Tensor<colsB>& tensorB)
{
    constexpr auto offcols = colsB % PACKAGE_LENGTH;

    const auto mask = remainderMaskSI<offcols>();

    Tensor<colsB, colsA> output;

    float* iterO;

    #pragma omp parallel for schedule(dynamic)
    for (::std::size_t c = 0; c < colsA; ++c)
    {
        iterO = (float*)output._values + c * colsB;

        PACKAGE_TYPE packageA = _LOAD1((float*)tensorA._values + c);

        ::std::size_t i = 0;

        for (; i + PACKAGE_LENGTH < colsB; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packageB = _LOAD((float*)tensorB._values + i);

            const PACKAGE_TYPE result = _MUL(packageA, packageB);

            _STORE(&iterO[i], result);
        }

        if constexpr (offcols)
        {
            const PACKAGE_TYPE packageB = _LOAD((float*)tensorB._values + i);

            const PACKAGE_TYPE result = _MUL(packageA, packageB);

            _MASKSTORE(&iterO[i], mask, result);
        }
    }

    return output;
}


// ------ TENSORS MASK -------


#ifdef __AVX2__

// Return a mask based on a given offset
// Only the offset values in the returned mask are set to 1, the others are set to 0
template<uint16_t offset>
inline static constexpr __m256i remainderMaskSI()
{
    static_assert(offset <= PACKAGE_LENGTH, "Error in remainderMask : offset is too big");
    // Make a mask of 8 bytes
    // No need to clip for missingLanes <= 8 because the shift is already good, results in zero
    uint64_t mask = ~(uint64_t) 0;
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

template<uint16_t offset>
inline static constexpr PACKAGE_TYPE remainderMask()
{
    return _CASTSI_PS(remainderMaskSI<offset>());
}