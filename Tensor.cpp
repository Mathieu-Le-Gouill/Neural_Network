#pragma once
#include "Tensor.h"
#include <chrono>
#include <random>
#include "debug.h"
#include <optional>
#include <sstream>
#include "compiler_optimizations.h"

// TODO : Look for OpenMP for threading and compare it with manual threading
//        See for the other implementation of the argmax function
//        See for the other implementation of the transpose function
//        Think about how the loop could be unrolled to use all the registers available
/*
* Something like that

        for (; b + 4 <= minibatchSize; b += 4)  // Unroll by 4 if possible
        {
            const PACKAGE_TYPE packedValuesB0 = _LOAD(tensor._values + (b + 0) * size + i);
            const PACKAGE_TYPE packedValuesB1 = _LOAD(tensor._values + (b + 1) * size + i);
            const PACKAGE_TYPE packedValuesB2 = _LOAD(tensor._values + (b + 2) * size + i);
            const PACKAGE_TYPE packedValuesB3 = _LOAD(tensor._values + (b + 3) * size + i);

            // Accumulate SIMD-packed results
            sum = _SUB(sum, packedValuesB0);
            sum = _SUB(sum, packedValuesB1);
            sum = _SUB(sum, packedValuesB2);
            sum = _SUB(sum, packedValuesB3);
        }

        // Process any remaining minibatch elements
        for (; b < minibatchSize; ++b)
        {
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);
            sum = _SUB(sum, packedValuesB);
        }


*/


// ------- TENSORS ALIAS -------


// Alias for the Tensor of two dimensions
template <std::size_t cols, std::size_t rows>
using Matrix = Tensor<cols, rows>;

// Alias for the Tensor of one dimension
template <std::size_t size>
using Vector = Tensor<size>;


// ------- TENSORS CONSTRUCTORS -------


// Initialize the tensor values
template<::std::size_t ...Dimensions>
inline void Tensor<Dimensions...>::init()
{
    _values = static_cast<float*>(_mm_malloc(_size * sizeof(float), PACKAGE_SIZE));
}


// Default constructor
template<std::size_t ...Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor()
{
    _values = nullptr;
}


// Copy constructor
template<std::size_t ...Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor(const Tensor<Dimensions...>& other)
{
    init();

    std::memcpy(this->_values, other._values, _size * sizeof(float));
}


// Rvalue assignement constructor
template <std::size_t... Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor(Tensor<Dimensions...>&& other) noexcept
{        
    this->_values = other._values;

    other._values = nullptr;
}


// Assignement constructor for a different dimensions Tensor of the same size
template <std::size_t... Dimensions>
template <std::size_t... OtherDimensions>
inline constexpr Tensor<Dimensions...>::Tensor(const Tensor<OtherDimensions...>& other)
{
    static_assert((1 * ... * Dimensions) == (1 * ... * OtherDimensions), "Error in Tensor move constructor: incorrect dimensions");

    init();

    std::memcpy(this->_values, other._values, _size * sizeof(float));
}


// Rvalue assignement constructor for a different dimensions Tensor of the same size
template <std::size_t... Dimensions>
template <std::size_t... OtherDimensions>
inline constexpr Tensor<Dimensions...>::Tensor(Tensor<OtherDimensions...>&& other) noexcept
{
    static_assert((1 * ... * Dimensions) == (1 * ... * OtherDimensions), "Error in Tensor move constructor: incorrect dimensions");

    this->_values = other._values;

    other._values = nullptr;
}


// Fill Tensors with a given value
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...>::Tensor(float value)
{
    // <!> TODO : Check if memset is not better
    init();

    const PACKAGE_TYPE packedValue = _SET1(value);

    size_t i = 0;

    UNROLL_LOOP(UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        _STORE(_values + i, packedValue);
    }

    if constexpr (_offset)
        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), packedValue);
}


// Fill the tensor from an initializer list
template<std::size_t ...Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor(std::initializer_list<float> values)
{
    debug_assert(values.size() == _size && "Error in Tensor constructor : the given initalizer list does not have a valid size.");

    init();

    std::memcpy(_values, values.begin(), _size * sizeof(float));
}



// Fill the tensor from an initializer list
template<std::size_t ...Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor(float values[_size])
{
    init();

    std::memcpy(_values, values, _size * sizeof(float));
}


// Fill Tensors with zeros
template <std::size_t... Dimensions>
Tensor<Dimensions...> zeros()
{
    Tensor<Dimensions...> output;

    output.init();

    const PACKAGE_TYPE packedValues = _SETZERO();

    constexpr uint16_t offset = output._offset;
    constexpr uint16_t size = output._size;

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        _STORE(output._values + i, packedValues);
    }

    if constexpr (offset)
        _MASKSTORE(output._values + i, remainderMaskSI<offset>(), packedValues);

    return output;
}


// Fill Tensors with ones
template <std::size_t... Dimensions>
inline Tensor<Dimensions...> ones()
{
    return Tensor<Dimensions...>(1.0f);
}


// Fill Tensors with random values from a normal distribution based on mean and the standard deviation
template <std::size_t... Dimensions>
Tensor<Dimensions...> normal(float mean, float std)
{
    Tensor<Dimensions...> output;

    output.init();

    constexpr uint16_t offset = output._offset;
    constexpr uint16_t size = output._size;

    auto seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();// To get differents epochs 
    std::default_random_engine generator(seed);// Create a generator of random numbers

    std::normal_distribution<float> distribution(mean, std);

    size_t i = 0;

    
    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE randomValues = _mm256_set_ps(distribution(generator), distribution(generator), distribution(generator), distribution(generator),
                                                  distribution(generator), distribution(generator), distribution(generator), distribution(generator));

        _STORE(output._values + i, randomValues);
    }

    if constexpr (offset)
    {
        const PACKAGE_TYPE randomValues = _mm256_set_ps(distribution(generator), distribution(generator), distribution(generator), distribution(generator),
            											distribution(generator), distribution(generator), distribution(generator), distribution(generator));
        
        _MASKSTORE(output._values + i, remainderMaskSI<offset>(), randomValues);
    }

    return output;
}


// Destructor
template<std::size_t ...Dimensions>
inline Tensor<Dimensions...>::~Tensor()
{
    if(_values)
        _mm_free(_values);  
}


// ------ TENSORS OPERATORS -------


// Copy operator with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
inline constexpr void Tensor<Dimensions...>::operator=(const Tensor<Dimensions...>& other)
{
    debug_assert(other._values && "Error in Tensor copy operator : an unitialized tensors is given as an argument.");

    if (this == &other)
		return;

    if(_values == nullptr)
        init();

    std::memcpy(this->_values, other._values, _size * sizeof(float));
}


// Copy operator with a rvalue Tensor of the same dimensions
template<std::size_t ...Dimensions>
inline constexpr void Tensor<Dimensions...>::operator=(Tensor<Dimensions...>&& other) noexcept
{
    if(_values)
        _mm_free(_values);
   
    this->_values = other._values;

    other._values = nullptr;
}


// Element-wise addition with a float
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(float value) const &
{
    Tensor<Dimensions...> output;

    output.init();

    const PACKAGE_TYPE packedValues = _SET1(value);

    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _STORE(output._values + i, _ADD(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _ADD(packedValuesA, packedValues));

    }

    return output;
}


// Element-wise subtraction with a float
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(float value) const &
{
    Tensor<Dimensions...> output;

    output.init();

    const PACKAGE_TYPE packedValues = _SET1(value);

    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _STORE(output._values + i, _SUB(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _SUB(packedValuesA, packedValues));

    }

    return output;
}


// Element-wise multiplication with a float
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(float value) const &
{
    Tensor<Dimensions...> output;

    output.init();

    const PACKAGE_TYPE packedValues = _SET1(value);

    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _STORE(output._values + i, _MUL(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValues)); // ------------------CHECK THIS------------------

    }

    return output;
}





// Element-wise division with a float
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(float value) const &
{
    Tensor<Dimensions...> output;

    output.init();

    const PACKAGE_TYPE packedValues = _SET1(value);

    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _STORE(output._values + i, _DIV(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
		const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

		_MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValues));

    }

    return output;
}


// Element-wise addition of a rvalue tensor with a float
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(float value) &&
{
    const PACKAGE_TYPE packedValues = _SET1(value);

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _STORE(this->_values + i, _ADD(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _ADD(packedValuesA, packedValues));

    }

    return *this;
}


// Element-wise subtraction of a rvalue tensor with a float
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(float value)&&
{
    const PACKAGE_TYPE packedValues = _SET1(value);

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _STORE(this->_values + i, _SUB(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _SUB(packedValuesA, packedValues));

    }

    return *this;
}


// Element-wise multiplication of a rvalue tensor with a float
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(float value)&&
{
    const PACKAGE_TYPE packedValues = _SET1(value);

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _STORE(this->_values + i, _MUL(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValues));

    }

    return *this;
}


// Element-wise division of a rvalue tensor with a float
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(float value)&&
{
    const PACKAGE_TYPE packedValues = _SET1(value);

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _STORE(this->_values + i, _DIV(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValues));

    }

    return *this;
}

// Element-wise addition with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(const Tensor<Dimensions...>& tensor) const & 
{
    Tensor<Dimensions...> output;

    output.init();

    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(output._values + i, _ADD(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
		const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
		const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i );

		_MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _ADD(packedValuesA, packedValuesB));

    }

    return output;
}


// Element-wise subtraction with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(const Tensor<Dimensions...>& tensor) const &
{
    Tensor<Dimensions...> output;

    output.init();

    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(output._values + i, _SUB(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _SUB(packedValuesA, packedValuesB));

    }

    return output;
}


// Element-wise multiplication with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(const Tensor<Dimensions...>& tensor) const &
{
    Tensor<Dimensions...> output;

    output.init();

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(output._values + i, _MUL(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValuesB));

    }

    return output;
}


// Element-wise division with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(const Tensor<Dimensions...>& tensor) const &
{
    Tensor<Dimensions...> output;

    output.init();

    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(output._values + i, _DIV(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValuesB));

    }

    return output;
}


// Element-wise addition with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(const Tensor<Dimensions...>& tensor) &&
{
    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _ADD(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _ADD(packedValuesA, packedValuesB));

    }

    return *this;
}


// Element-wise subtraction with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(const Tensor<Dimensions...>& tensor) &&
{
    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _SUB(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _SUB(packedValuesA, packedValuesB));

    }

    return *this;
}


// Element-wise multiplication with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(const Tensor<Dimensions...>& tensor) &&
{
    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _MUL(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValuesB));

    }

    return *this;
}


// Element-wise division with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(const Tensor<Dimensions...>& tensor) &&
{
    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _DIV(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValuesB));

    }

    return *this;
}


// Element-wise addition with a rvalue Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(Tensor<Dimensions...>&& tensor) const &
{
    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _ADD(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _ADD(packedValuesA, packedValuesB));

    }

    return tensor;
}


// Element-wise subtraction with a rvalue Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(Tensor<Dimensions...>&& tensor) const &
{
    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _SUB(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _SUB(packedValuesA, packedValuesB));

    }

    return tensor;
}


// Element-wise multiplication with a rvalue Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(Tensor<Dimensions...>&& tensor) const &
{
    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _MUL(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValuesB));

    }

    return tensor;
}


// Element-wise division with a rvalue Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(Tensor<Dimensions...>&& tensor) const &
{
    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _DIV(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValuesB));

    }

    return tensor;
}


template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(Tensor<Dimensions...>&& tensor)&&
{
    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _ADD(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _ADD(packedValuesA, packedValuesB));
    }

    return tensor;
}


template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(Tensor<Dimensions...>&& tensor)&&
{
    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _SUB(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _SUB(packedValuesA, packedValuesB));

    }

    return tensor;
}


template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(Tensor<Dimensions...>&& tensor)&&
{
    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _MUL(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValuesB));

    }

    return tensor;
}


template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(Tensor<Dimensions...>&& tensor)&&
{
    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _DIV(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValuesB));

    }

    return tensor;
}

// Element-wise addition of the tensor with another tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator+=(const Tensor<Dimensions...>& tensor)
{
    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _ADD(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _ADD(packedValuesA, packedValuesB));
    }
}


// Element-wise subtraction of the tensor with another tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator-=(const Tensor<Dimensions...>& tensor)
{
    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _SUB(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _SUB(packedValuesA, packedValuesB));
    }
}


// Element-wise multiplication of the tensor with another tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator*=(const Tensor<Dimensions...>& tensor)
{
    size_t i = 0;
    
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _MUL(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValuesB));
    }
}


// Element-wise division of the tensor with another tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator/=(const Tensor<Dimensions...>& tensor)
{
    size_t i = 0;
        
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _DIV(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValuesB));
    }
}

template<std::size_t ...Dimensions>
constexpr float& Tensor<Dimensions...>::operator()(std::size_t index) const
{
    debug_assert(index < _size && "Error in Tensor operator() : the given index is out of bounds.");

    return _values[index];

}

template<std::size_t ...Dimensions>
constexpr bool Tensor<Dimensions...>::operator==(const Tensor<Dimensions...>& tensor) const
{
    if (this == &tensor)
		return true;

	if (_size != tensor._size)
		return false;

    for (size_t i = 0; i < _size; ++i)
    {
		if (_values[i] != tensor._values[i])
			return false;
	}

	return true;
}


// ------ TENSORS BATCHES OPERATORS -------


// Element-wise addition with each tensors batches
template<std::size_t... Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator+(const Tensor<Dimensions..., batch_size>& tensor) const
{
    Tensor<Dimensions..., batch_size> output;

    output.init();

    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t b, i;
    
    for (b = 0; b < batch_size; ++b)
    {
        for (i = 0; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(output._values + b * size + i, _ADD(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(output._values + b * size + i, remainderMaskSI<offset>(), _ADD(packedValuesA, packedValuesB));

        }
    }

    return output;
}


// Element-wise subtraction with each tensors batches
template<std::size_t... Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator-(const Tensor<Dimensions..., batch_size>& tensor) const
{
    Tensor<Dimensions..., batch_size> output;

    output.init();

    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t b, i;

    for (b = 0; b < batch_size; ++b)
    {
        for (i = 0; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(output._values + b * size + i, _SUB(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(output._values + b * size + i, remainderMaskSI<offset>(), _SUB(packedValuesA, packedValuesB));

        }
    }

    return output;
}


// Element-wise multiplication with each tensors batches
template<std::size_t... Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator*(const Tensor<Dimensions..., batch_size>& tensor) const
{
    Tensor<Dimensions..., batch_size> output;

    output.init();

    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t b, i;

    for (b = 0; b < batch_size; ++b)
    {
        for (i = 0; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(output._values + b * size + i, _MUL(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(output._values + b * size + i, remainderMaskSI<offset>(), _MUL(packedValuesA, packedValuesB));

        }
    }

    return output;
}


// Element-wise division with each tensors batches
template<std::size_t... Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator/(const Tensor<Dimensions..., batch_size>& tensor) const
{
    Tensor<Dimensions..., batch_size> output;

    output.init();

    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t b, i;

    for (b = 0; b < batch_size; ++b)
    {
        for (i = 0; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(output._values + b * size + i, _DIV(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(output._values + b * size + i, remainderMaskSI<offset>(), _DIV(packedValuesA, packedValuesB));

        }
    }

    return output;
}


// Element-wise addition with each tensors batches rvalue Tensor
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator+(Tensor<Dimensions..., batch_size>&& tensor) const
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;
    
    size_t b, i;

    for (b = 0; b < batch_size; ++b)
    {
        for (i = 0; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(tensor._values + b * size + i, _ADD(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(tensor._values + b * size + i, remainderMaskSI<offset>(), _ADD(packedValuesA, packedValuesB));

        }
    }

    return tensor;
}


// Element-wise subtraction with each tensors batches rvalue Tensor
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator-(Tensor<Dimensions..., batch_size>&& tensor) const
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t b, i;

    for (b = 0; b < batch_size; ++b)
    {
        for (i = 0; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(tensor._values + b * size + i, _SUB(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(tensor._values + b * size + i, remainderMaskSI<offset>(), _SUB(packedValuesA, packedValuesB));

        }
    }

    return tensor;
}


// Element-wise multiplication with each tensors batches rvalue Tensor
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator*(Tensor<Dimensions..., batch_size>&& tensor) const
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t b, i;

    for (b = 0; b < batch_size; ++b)
    {
        for (i = 0; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(tensor._values + b * size + i, _MUL(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(tensor._values + b * size + i, remainderMaskSI<offset>(), _MUL(packedValuesA, packedValuesB));

        }
    }

    return tensor;
}


// Element-wise division with each tensors batches rvalue Tensor
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator/(Tensor<Dimensions..., batch_size>&& tensor) const
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t b, i;

    for (b = 0; b < batch_size; ++b)
    {
        
        for (i = 0; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(tensor._values + b * size + i, _DIV(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(tensor._values + b * size + i, remainderMaskSI<offset>(), _DIV(packedValuesA, packedValuesB));

        }
    }

    return tensor;
}


// Element-wise addition of the tensor with another tensor of the same dimensions
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr void Tensor<Dimensions...>::operator+=(const Tensor<Dimensions..., batch_size>& tensor)
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        PACKAGE_TYPE sum = packedValuesA;

        for (size_t b = 0; b < batch_size; ++b)
        {
			const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

			sum = _ADD(sum, packedValuesB);
		}

        _STORE(this->_values + i, sum);
    }

    if constexpr (offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        PACKAGE_TYPE sum = packedValuesA;

        
        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            sum = _ADD(sum, packedValuesB);
        }

        _MASKSTORE(this->_values + i, remainderMaskSI<offset>(), sum);
    }
}


// Element-wise subtraction of the tensor with each tensors batches
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr void Tensor<Dimensions...>::operator-=(const Tensor<Dimensions..., batch_size>& tensor)
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        PACKAGE_TYPE sum = packedValuesA;

        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            sum = _SUB(sum, packedValuesB);
        }

        _STORE(this->_values + i, sum);
    }

    if constexpr (offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        PACKAGE_TYPE sum = packedValuesA;


        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            sum = _SUB(sum, packedValuesB);
        }

        _MASKSTORE(this->_values + i, remainderMaskSI<offset>(), sum);
    }
}


// Sum of Element-wise multiplication of the tensor with each tensors batches
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr void Tensor<Dimensions...>::operator*=(const Tensor<Dimensions..., batch_size>& tensor)
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t i = 0;
    size_t b;

    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        PACKAGE_TYPE sum = _SETZERO();

        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        for (b = 0; b < batch_size; ++b)
        {
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            const PACKAGE_TYPE product = _MUL(packedValuesA, packedValuesB);

            sum = _ADD(sum, product);
        }

        _STORE(this->_values + i, sum);
    }

    if constexpr (offset)
    {
        PACKAGE_TYPE sum = _SETZERO();

        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            const PACKAGE_TYPE product = _MUL(packedValuesA, packedValuesB);
            sum = _ADD(sum, product);
        }

        _MASKSTORE(this->_values + i, remainderMaskSI<offset>(), sum);
    }
}


// Sum of Element - wise division of the tensor with each tensors batches
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr void Tensor<Dimensions...>::operator/=(const Tensor<Dimensions..., batch_size>& tensor)
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t i = 0;
    size_t b;

    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        PACKAGE_TYPE sum = _SETZERO();

        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        for (b = 0; b < batch_size; ++b)
        {
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            const PACKAGE_TYPE product = _DIV(packedValuesA, packedValuesB);

            sum = _ADD(sum, product);
        }

        _STORE(this->_values + i, sum);
    }

    if constexpr (offset)
    {
        PACKAGE_TYPE sum = _SETZERO();

        const PACKAGE_TYPE packedValuesA = _LOAD(this->_values + i);

        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_TYPE packedValuesB = _LOAD(tensor._values + b * size + i);

            const PACKAGE_TYPE product = _DIV(packedValuesA, packedValuesB);
            sum = _ADD(sum, product);
        }

        _MASKSTORE(this->_values + i, remainderMaskSI<offset>(), sum);
    }
}


// ------ TENSORS BASIC FUNCTIONS -------


// To print the tensor elements
template <std::size_t... Dimensions>
void print(const Tensor<Dimensions...>& tensor, float decimals)
{
    assert(tensor._values != nullptr && "Error: the given tensor is empty.");

    std::cout << std::fixed << std::setprecision(static_cast<int>(decimals));

    constexpr std::size_t dimensions[] = { Dimensions... };
    const size_t size = tensor._size;

    if constexpr (sizeof...(Dimensions) == 1) {
        // For 1D tensor (vector)
        for (std::size_t d = 0; d < size; ++d) {
            std::cout << tensor._values[d] << " ";
        }
    }
    else if constexpr (sizeof...(Dimensions) == 2) {
        // For 2D tensor (matrix)
        const size_t rows = dimensions[1];
        const size_t cols = dimensions[0];
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                std::cout << tensor._values[r * cols + c] << " ";
            }
            std::cout << std::endl;
        }
    }
    else {
        // For Higher-dimensional tensor
        const size_t rows = dimensions[1];
        const size_t cols = dimensions[0];
        for (std::size_t d = 0; d < size; ++d) {
            std::cout << tensor._values[d] << " ";

            if ((d + 1) % cols == 0) {
                std::cout << std::endl;
            }

            if ((d + 1) % (cols * rows) == 0 && d != size - 1) {
                std::cout << std::endl;
            }
        }
    }

    std::cout << std::defaultfloat;
}


// Compute the absolute value of each element in the tensor
template<std::size_t ...Dimensions>
Tensor<Dimensions...> abs(const Tensor<Dimensions...>& tensor)
{
    Tensor<Dimensions...> output;

    output.init();

    constexpr uint16_t offset = tensor._offset;
    constexpr uint16_t size = tensor._size;

    auto minus1 = _SET1_EPI32(-1);
    const PACKAGE_TYPE absMask = _CASTSI_PS(_mm256_srli_epi32(minus1, 1));

    size_t i = 0;

    
    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValues = _LOAD(tensor._values + i);

        _STORE(output._values + i, _AND(absMask, packedValues));// Clear the sign bit
    }

    if constexpr (offset)
    {
        const PACKAGE_TYPE packedValues = _LOAD(tensor._values + i);

		_MASKSTORE(output._values + i, remainderMaskSI<offset>(), _AND(absMask, packedValues));
    }

    return output;
}


// To get the shape of the tensor
template<std::size_t ...Dimensions>
constexpr std::string Tensor<Dimensions...>::shape() const
{
    std::stringstream ss;

    ss << "(";
    ((ss << Dimensions << ", "), ...);
    std::string result = ss.str();
    result.pop_back();
    result.back() = ')';

    return result;
}

// Reshape the dimensions of the tensor to a compatible one
template<std::size_t ...Dimensions>
template<std::size_t ...newDimensions>
constexpr Tensor<newDimensions...> Tensor<Dimensions...>::reshape() const &
{
    static_assert(_size == (1 * ... * newDimensions), "Error in Tensor reshape : incorrect dimensions");

    return Tensor<newDimensions...>(*this);
}


// Reshape the dimensions of the tensor to a compatible one for rvalue tensor
template<::std::size_t ...otherDimensions>
template<std::size_t ...newDimensions>
constexpr Tensor<newDimensions...> Tensor<otherDimensions...>::reshape() &&
{
    static_assert(_size == (1 * ... * newDimensions), "Error in Tensor reshape : incorrect dimensions");

    Tensor<newDimensions...> output(std::move(*this));

    return output;
}


// Split the tensor into multiple tensors of the given dimensions
template<std::size_t ...Dimensions>
template<std::size_t ...splittedDimensions>
constexpr Tensor<splittedDimensions...>* Tensor<Dimensions...>::split()
{
    constexpr size_t splittedSize = Tensor<splittedDimensions...>::_size;
    constexpr size_t numSplits = this->_size / splittedSize;

    static_assert(this->_size % splittedSize == 0, "Error in Tensor split : incorrect dimensions");
    debug_assert(this->_values != nullptr && "Error in Tensor split : tensor not initialized");

    Tensor<splittedDimensions...>* splittedTensors = new Tensor<splittedDimensions...>[numSplits];

    
    for (size_t i = 0; i < numSplits; ++i)
    {
        splittedTensors[i]._values = this->_values + i * splittedSize;
    }

    return splittedTensors;
}

// Flatten the tensor to a one dimension tensor
template<std::size_t ...Dimensions>
constexpr Tensor<(1 * ... * Dimensions)> Tensor<Dimensions...>::flatten() const &
{
    if constexpr (sizeof...(Dimensions) == 1)
        return *this;

    return Tensor<_size>(*this);   
}

// Flatten the tensor to a one dimension tensor for rvalue tensor
template<std::size_t ...Dimensions>
constexpr Tensor<(1 * ... * Dimensions)> Tensor<Dimensions...>::flatten() &&
{
    if constexpr (sizeof...(Dimensions) == 1)
        return *this;

    return Tensor<_size>(std::move(*this));
}


// Compute the sum of each values in the tensor
template<std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::sum() const
{
    PACKAGE_TYPE packedSum;

    if constexpr (_size >= PACKAGE_LENGTH)
    {
        packedSum = _LOAD(_values);

        size_t i = PACKAGE_LENGTH;

        
        for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValues = _LOAD(_values + i);

            packedSum = _ADD(packedValues, packedSum);
        }

        if constexpr (_offset)
        {
            const PACKAGE_TYPE packedValues = _LOAD(_values + i);

            packedSum = _ADD( _AND(packedValues, remainderMask<_offset>()), packedSum);
        }
    }
    else
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values);

        packedSum = _AND(packedValues, remainderMask<_size>());
    }
       
    const float sum = horizontal_sum8(packedSum);

    return sum;
}


// Find the index of the maximum value in the tensor
// ref : https://en.algorithmica.org/hpc/algorithms/argmin/
template<std::size_t ...Dimensions>
constexpr size_t Tensor<Dimensions...>::argmax() const
{
    PACKAGE_TYPE maxValues;

    size_t argmax = 0;
    float max;

    if constexpr (_size >= PACKAGE_LENGTH)
    {
        max = horizontal_max8(_LOAD(_values));
        maxValues = _SET1(max);

        size_t i = PACKAGE_LENGTH;

        for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValues = _LOAD(_values + i);

            maxValues = _MAX(packedValues, maxValues);

            auto compMask = _CMPGT_EPI32( _CASTPS_SI(maxValues), _CASTPS_SI(packedValues));

            if (!_mm256_testz_si256(compMask, compMask)) [[unlikely]]
            {
                max = horizontal_max8(maxValues);
                argmax = i;

                maxValues = _SET1(max);
            }
				
        }

        if constexpr (_offset)
        {

            for (; i < _size; ++i)
            {
                if (_values[i] > max)
                {
                    max = _values[i];
                    argmax = i;

                    for (size_t r = i+1; r < _size; ++r)
                    {
                        argmax = (_values[r] > _values[argmax]) ? r : argmax;
                    }

                    return argmax;
                }
            }
        }
    }
    else
	{
        max = _values[0];
         
        for (uint16_t i = 1; i < _size; ++i)
        {
            if (_values[i] > max)
            {
                max = _values[i];
                argmax = i;
            }
        }

        return argmax;
	}

    
    for (size_t i = argmax; i < argmax + PACKAGE_LENGTH; ++i)
    {
        if (_values[i] == max)
            return i;
	}

	return -1;
}


// Find the maximum value in the tensor
template<std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::max() const
{
    PACKAGE_TYPE maxValues;

    if constexpr (_size >= PACKAGE_LENGTH)
    {
        maxValues = _LOAD(_values);

        size_t i = PACKAGE_LENGTH;

        for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValues = _LOAD(_values + i);

            maxValues = _MAX(packedValues, maxValues);
        }

        if constexpr (_offset)
        {
            PACKAGE_TYPE minimum = _SET1(-FLT_MAX);
            constexpr int mask = (1 << _offset) - 1;

            const PACKAGE_TYPE packedValues = _LOAD(_values + i);

            maxValues = _MAX(_BLEND(minimum, packedValues, mask), maxValues);
        }
    }
    else
    {
        PACKAGE_TYPE minimum = _SET1(-FLT_MAX);
        constexpr int mask = (1 << _size) - 1;

        const PACKAGE_TYPE packedValues = _LOAD(_values);

        maxValues = _BLEND(minimum, packedValues, mask);
    }
       

    float max = horizontal_max8(maxValues);

    return max;
}


// Compute the mean of the values in the tensor
template<std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::mean() const
{
    const float mean = sum() / (float)_size;

    return mean;
}


// Compute the variance of the values in the tensor based on a given mean
template<std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::variance(float mean) const
{
    PACKAGE_TYPE packedMean = _SET1(mean);

    PACKAGE_TYPE deviation;

    if constexpr (_size >= PACKAGE_LENGTH)
    {
        deviation = _SUB( _LOAD(_values), packedMean);

        size_t i = PACKAGE_LENGTH;


        
        for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packedValues = _LOAD(_values + i);

            deviation = _ADD( _SUB(packedValues, packedMean), deviation);
        }

        if constexpr (_offset)
        {
            const PACKAGE_TYPE packedValues = _SUB( _LOAD(_values + i), packedMean);

            const PACKAGE_TYPE maskedValues = _AND(packedValues, remainderMask<_offset>());

            deviation = _ADD( maskedValues, deviation);
        }
    }
    else
    {
        const PACKAGE_TYPE packedValues = _LOAD(_values);

        deviation = _AND( _SUB(packedValues, packedMean), remainderMask<_size>());
    }
        

    const float variance = horizontal_sum8( _MUL(deviation, deviation)) / (float) _size;

    return variance;
}


// ------ TENSORS MATRIX OPERATIONS -------


// Element-wise addition between tensorC and the element-wise multiplication of tensorA and tensorB
// @return tensorA * tensorB + tensorC
template<std::size_t ...Dimensions>
Tensor<Dimensions...> multiply_and_add(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC)
{
    Tensor<Dimensions...> output;
    output.init();

    constexpr size_t size = output._size;
    constexpr uint16_t offset = output._offset;

    size_t i = 0;

    
    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(tensorA._values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensorB._values + i);
        const PACKAGE_TYPE packedValuesC = _LOAD(tensorC._values + i);

        const PACKAGE_TYPE result = _FMADD(packedValuesA, packedValuesB, packedValuesC);

        _STORE(output._values + i, result);
    }

    if constexpr (offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(tensorA._values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensorB._values + i);
        const PACKAGE_TYPE packedValuesC = _LOAD(tensorC._values + i);

        const PACKAGE_TYPE result = _FMADD(packedValuesA, packedValuesB, packedValuesC);

        _MASKSTORE(output._values + i, remainderMaskSI<offset>(), result);
    }

    return output;
}


// Element-wise subtraction between tensorC and the element-wise multiplication of tensorA and tensorB
// @return tensorA * tensorB - tensorC
template<std::size_t ...Dimensions>
Tensor<Dimensions...> multiply_and_sub(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC)
{
    Tensor<Dimensions...> output;
    output.init();

    constexpr size_t size = output._size;
    constexpr uint16_t offset = output._offset;

    size_t i = 0;

    
    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(tensorA._values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensorB._values + i);
        const PACKAGE_TYPE packedValuesC = _LOAD(tensorC._values + i);

        const PACKAGE_TYPE result = _FMSUB(packedValuesA, packedValuesB, packedValuesC);

        _STORE(output._values + i, result);
    }

    if constexpr (offset)
    {
        const PACKAGE_TYPE packedValuesA = _LOAD(tensorA._values + i);
        const PACKAGE_TYPE packedValuesB = _LOAD(tensorB._values + i);
        const PACKAGE_TYPE packedValuesC = _LOAD(tensorC._values + i);

        const PACKAGE_TYPE result = _FMSUB(packedValuesA, packedValuesB, packedValuesC);

        _MASKSTORE(output._values + i, remainderMaskSI<offset>(), result);
    }

    return output;
}


// Transpose the tensor
template <std::size_t cols, std::size_t rows, std::size_t... rest>
Tensor<rows, cols, rest...> transpose(const Tensor<cols, rows, rest...>& tensor)
{
    Tensor<rows, cols, rest...> output;
    output.init();

    constexpr std::size_t colsA = cols;
    constexpr std::size_t rowsA = rows;

    constexpr std::size_t colsB = rows;
    constexpr std::size_t rowsB = cols;

    constexpr std::uint16_t offColsA = colsA % PACKAGE_LENGTH;
    constexpr std::uint16_t offRowsA = rowsA % PACKAGE_LENGTH;

    float* valuesA = tensor._values;
    float* valuesB = output._values;

    std::size_t r = 0;

    
    for (; r + PACKAGE_LENGTH <= rowsA; r += PACKAGE_LENGTH)
    {
        std::size_t c = 0;

        for (; c + PACKAGE_LENGTH <= colsA; c += PACKAGE_LENGTH)
        {
            PACKAGE_TYPE row[PACKAGE_LENGTH];

            for (std::uint16_t i = 0; i < PACKAGE_LENGTH; ++i)
            {
				row[i] = _LOAD(tensor._values + (r + i) * colsA + c);

            }

            #ifdef PACKAGE_M256
            transpose8(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]);

            #elif defined PACKAGE_M128
            _MM_TRANSPOSE4_PS(row[0], row[1], row[2], row[3]);
            #endif

            for(std::uint16_t i = 0; i < PACKAGE_LENGTH; ++i)
			{
                _STORE(output._values + (c + i) * colsB + r, row[i]);
			}
        }

        if constexpr (offColsA)
        {
            PACKAGE_TYPE row[PACKAGE_LENGTH];

            for (std::uint16_t i = 0; i < PACKAGE_LENGTH; ++i)
            {
                row[i] = _LOAD(tensor._values + (r + i) * colsA + c);
            }

            #ifdef PACKAGE_M256
            transpose8(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]);

            #elif defined PACKAGE_M128
            _MM_TRANSPOSE4_PS(row[0], row[1], row[2], row[3]);
            #endif

            for (std::uint16_t i = 0; i < offColsA; ++i)
            {
                _STORE(output._values + (c + i) * colsB + r, row[i]);
            }
        }
    }

    if constexpr (offRowsA)
    {
        std::size_t c = 0;

        
        for (; c + PACKAGE_LENGTH <= colsA; c += PACKAGE_LENGTH)
        {
            PACKAGE_TYPE row[PACKAGE_LENGTH];

            for (std::uint16_t i = 0; i < offRowsA; ++i)
            {
                row[i] = _LOAD(tensor._values + (r + i) * colsA + c);
            }


            #ifdef PACKAGE_M256
            transpose8(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]);

            #elif defined PACKAGE_M128
            _MM_TRANSPOSE4_PS(row[0], row[1], row[2], row[3]);
            #endif

            for (std::uint16_t i = 0; i < PACKAGE_LENGTH; ++i)
            {
                _MASKSTORE(output._values + (c + i) * colsB + r, remainderMaskSI<offRowsA>(), row[i]);
            }
        }

        if constexpr (offColsA)
        {
            PACKAGE_TYPE row[PACKAGE_LENGTH];

            for (std::uint16_t i = 0; i < offRowsA; ++i)
            {
                row[i] = _LOAD(tensor._values + (r + i) * colsA + c);
            }

            #ifdef PACKAGE_M256
            transpose8(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]);

            #elif defined PACKAGE_M128
            _MM_TRANSPOSE4_PS(row[0], row[1], row[2], row[3]);
            #endif

            for (std::uint16_t i = 0; i < offColsA; ++i)
            {
                _MASKSTORE(output._values + (c + i) * colsB + r, remainderMaskSI<offRowsA>(), row[i]);
            }
        }
    }
   
    return output;
}


// ------ TENSORS MATRIX MULTIPLICATION -------


// Matrix multiplication between tensorA and tensorB
template<std::size_t colsA, std::size_t rowsA, std::size_t colsB, std::size_t... rest>
Tensor<colsB, rowsA, rest...> mul(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsB, colsA, rest...>& tensorB)
{
    constexpr auto offCols = colsB % PACKAGE_LENGTH;
    constexpr std::size_t outerLength = (1 * ... * rest);

    const auto mask = remainderMaskSI<offCols>();

    Tensor<colsB, rowsA, rest...> output;

    output.init();
    
    const float* iterA = tensorA._values;
    const float* iterB;

    float* iterO = output._values;

    
    for (std::size_t l = 0; l < outerLength; ++l)
    {
        for (std::size_t r = 0; r < rowsA; ++r)
        {
            std::size_t c = 0;

            for (; c + PACKAGE_LENGTH <= colsB; c += PACKAGE_LENGTH)
            {
                iterB = tensorB._values + c + l * colsB * colsA;

                PACKAGE_TYPE sum = _MUL(_LOAD1(iterA), _LOAD(iterB));

                for (std::size_t i = 1; i < colsA; ++i)
                {
                    const PACKAGE_TYPE packageA = _LOAD1(iterA + i);
                    const PACKAGE_TYPE packageB = _LOAD(iterB + i * colsB);

                    sum = _FMADD(packageA, packageB, sum);
                }

                _STORE(iterO + c, sum);
            }

            if constexpr (offCols)
            {
                iterB = tensorB._values + c + l * colsB * colsA;

                PACKAGE_TYPE sum = _MUL(_LOAD1(iterA), _LOAD(iterB));

                for (std::size_t i = 1; i < colsA; ++i)
                {
                    const PACKAGE_TYPE packageA = _LOAD1(iterA + i);
                    const PACKAGE_TYPE packageB = _LOAD(iterB + i * colsB);

                    sum = _FMADD(packageA, packageB, sum);
                }

                _MASKSTORE(iterO + c, mask, sum);
            }

            iterA += colsA;
            iterO += colsB;
        }
    }

    return output;
}


// Matrix multiplication between tensorA and the transpose of tensorB
template<std::size_t colsA, std::size_t rowsA, std::size_t colsB, std::size_t... rest>
Tensor<colsB, rowsA, rest...> mul_transposed(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsA, colsB, rest...>& tensorB)
{
    constexpr auto offcols = colsA % PACKAGE_LENGTH;
    constexpr std::size_t outerLenght = (1 * ... * rest);

    const auto mask = remainderMaskSI<offcols>();

    Tensor<colsB, rowsA, rest...> output;

    output.init();

    float* iterA;
    float* iterB;

    float* iterO = output._values;

    
    for (std::size_t l = 0; l < outerLenght; ++l)
    {
        for (std::size_t r = 0; r < rowsA; ++r)
        {
            for (std::size_t c = 0; c < colsB; ++c)
            {
                iterA = tensorA._values + colsA * r + colsA * rowsA * l;
                iterB = tensorB._values + colsA * c + colsA * colsB * l;

                PACKAGE_TYPE sum;

                if constexpr (colsA >= PACKAGE_LENGTH)
                {
                    sum = _MUL(_LOAD(iterA), _LOAD(iterB));

                    std::size_t i = PACKAGE_LENGTH;

                    for (; i + PACKAGE_LENGTH <= colsA; i += PACKAGE_LENGTH)
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
                {
                    const PACKAGE_TYPE packageA = _LOAD(iterA);
                    const PACKAGE_TYPE packageB = _LOAD(iterB);

                    sum = _AND(_MUL(packageA, packageB), remainderMask<colsA>());
                }

                *iterO = horizontal_sum8(sum);

                ++iterO;
            }
        }
    }

    return output;
}


// Matrix multiplication between tensorA and the transpose of tensorB as a scalar
template<std::size_t colsA, std::size_t rowsA>
Tensor<rowsA> mul_b_transposed_scalar(const Tensor<colsA, rowsA>& tensorA, const Tensor<colsA>& tensorB)
{
    constexpr auto offCols = colsA % PACKAGE_LENGTH;

    const auto mask = remainderMaskSI<offCols>();

    Tensor<rowsA> output;

    output.init();

    float* iterA;
    float* iterB = tensorB._values;

    float* iterO = output._values;

    
    for (std::size_t r = 0; r < rowsA; ++r)
    {
        iterA = tensorA._values + r * colsA;

        PACKAGE_TYPE sum;

        if constexpr (colsA >= PACKAGE_LENGTH)
        {
            sum = _MUL( _LOAD(iterA), _LOAD(iterB));

            std::size_t i = PACKAGE_LENGTH;

            for (; i + PACKAGE_LENGTH <= colsA; i += PACKAGE_LENGTH)
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
        {
            const PACKAGE_TYPE packageA = _LOAD(iterA);
            const PACKAGE_TYPE packageB = _LOAD(iterB);

            sum = _AND( _MUL(packageA, packageB), remainderMask<colsA>());
        }

        *iterO = horizontal_sum8(sum);

        ++iterO;
    }

    return output;
}


// Matrix multiplication between the transpose of tensorA and tensorB both as a scalar
template<std::size_t colsA, std::size_t colsB>
Tensor<colsB, colsA> mul_transposed_scalar(const Tensor<colsA>& tensorA, const Tensor<colsB>& tensorB)
{
    constexpr auto offset = colsB % PACKAGE_LENGTH;

    const auto mask = remainderMaskSI<offset>();

    Tensor<colsB, colsA> output;

    output.init();

    
    for (std::size_t c = 0; c < colsA; ++c)
    {
        PACKAGE_TYPE packageA = _LOAD1(tensorA._values + c);

        std::size_t i = 0;

        for (; i + PACKAGE_LENGTH <= colsB; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packageB = _LOAD(tensorB._values + i);

            const PACKAGE_TYPE result = _MUL(packageA, packageB);

            _STORE(output._values + i + c * colsB, result);
        }

        if constexpr (offset)
        {
            const PACKAGE_TYPE packageB = _LOAD(tensorB._values + i);

            const PACKAGE_TYPE result = _MUL(packageA, packageB);

            _MASKSTORE(output._values + i + c * colsB, mask, result);
        }
    }

    
    for (std::size_t c = 0; c < colsA; ++c)
    {
        PACKAGE_TYPE packageA = _LOAD1(tensorA._values + c);

        std::size_t i = 0;

        for (; i + PACKAGE_LENGTH <= colsB; i += PACKAGE_LENGTH)
        {
            const PACKAGE_TYPE packageB = _LOAD(tensorB._values + i);
            const PACKAGE_TYPE packageC = _LOAD(output._values + i + c * colsB);

            const PACKAGE_TYPE result = _FMADD(packageA, packageB, packageC);

            _STORE(output._values + i + c * colsB, result);
        }

        if constexpr (offset)
        {
            const PACKAGE_TYPE packageB = _LOAD(tensorB._values + i);
            const PACKAGE_TYPE packageC = _LOAD(output._values + i + c * colsB);

            const PACKAGE_TYPE result = _FMADD(packageA, packageB, packageC);

            _MASKSTORE(output._values + i + c * colsB, mask, result);
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