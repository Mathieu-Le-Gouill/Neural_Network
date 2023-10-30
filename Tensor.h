#pragma once
#include <iostream>
#include "simd_utils.h"
#include "debug.h"
#include <random>

template <size_t... Dimensions>
class Tensor {

    static constexpr size_t _size = (1 * ... * Dimensions);

    static_assert(_size > 0 && "Tensor size must be > 0");

    static constexpr uint16_t _offset = _size % PACKAGE_LENGTH;
    static constexpr size_t _numPackages = _size / PACKAGE_LENGTH;

public:

    // Default constructor
    constexpr Tensor() {}

    // Copy constructor
    constexpr Tensor(const Tensor<Dimensions...>& other)
    {
        *this = other;
    }

    // Rvalue assignement constructor
    constexpr Tensor(Tensor<Dimensions...>&& other) noexcept
    {
        this->_begin = other._begin;
        this->_end = other._end;

        other._begin = nullptr;
    }

    // Rvalue assignement constructor
    template<size_t... otherDimensions>
    constexpr Tensor(Tensor<otherDimensions...>&& other) noexcept
    {
        static_assert((1 * ... * Dimensions) == (1 * ... * otherDimensions), "Error in Tensor move constructor : incorrect dimensions");
        this->_begin = other._begin;
        this->_end = other._end;

        other._begin = nullptr;
    }


    // Fill constructor
    constexpr Tensor(float value)
    {
        const PACKAGE_TYPE packedValue = _SET1(value);
        PACKAGE_TYPE* iter = _begin;

        while (iter < _end)
        {
            *iter = packedValue;

            ++iter;
        }

        if constexpr (_offset)
            _MASKSTORE((float*)iter, remainderMask<_offset>(), packedValue);
    }


    constexpr Tensor(std::initializer_list<float> values)
    {
        debug_assert(values.size() == _size && "Error in Tensor constructor : the given initalizer list does not have a valid size.");

        std::memcpy(_begin, values.begin(), _size * sizeof(float));
    }


    // Destructor
    ~Tensor() {
        if(_begin)
            _mm_free(_begin);
    }

    // Copy operator with same dimensions Tensor
    constexpr void operator= (const Tensor<Dimensions...>& other)
    {
        std::memcpy(_begin, other._begin, _size * sizeof(float));
    }

    // Assignement operator of same dimensions rvalue Tensor
    constexpr void operator= (Tensor<Dimensions...>&& other) noexcept
    {
        this->_begin = other._begin;
        this->_end = other._end;

        other._begin = nullptr;
    }

    // Access operator to get a reference to the element at a given set of indices
    /*constexpr float& operator()(size_t indices...)
    {
        const size_t index = calculateFlatIndex(indices);
        debug_assert(index < _size && "Error in Tensor operator(): the given indices are out of bound.");
        return ((float*)_begin)[index];
    }*/

    template<size_t... newDimensions>
    Tensor<newDimensions...> reshape()
    {
        static_assert(_size == (1 * ... * newDimensions), "Error in Tensor reshape : incorrect dimensions");

       Tensor<newDimensions...> output(std::move(*this));
       return output;
    }


    Tensor<_size> flatten()
    {
        Tensor<_size> output(std::move(*this));
        return output;
    }


    friend void print(const Tensor<Dimensions...>& tensor)
    {
        float* iter = (float*)tensor._begin;
        size_t dimensions[] = { Dimensions... };

        for (size_t d = 0; d < _size; ++d) {
            std::cout << *iter << " ";
            ++iter;

            if ((d + 1) % dimensions[0] == 0) {
                std::cout << std::endl;
            }
        }
    }

    // Element-wise addition with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator+(const Tensor<Dimensions...>& tensor)
    {
        Tensor<Dimensions...> output;

        const PACKAGE_TYPE* iterA = this->_begin;
        const PACKAGE_TYPE* iterB = tensor._begin;

        PACKAGE_TYPE* iterO = output._begin;

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
    constexpr Tensor<Dimensions...> operator-(const Tensor<Dimensions...>& tensor)
    {
        Tensor<Dimensions...> output;

        const PACKAGE_TYPE* iterA = this->_begin;
        const PACKAGE_TYPE* iterB = tensor._begin;

        PACKAGE_TYPE* iterO = output._begin;

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
    constexpr Tensor<Dimensions...> operator*(const Tensor<Dimensions...>& tensor)
    {
        Tensor<Dimensions...> output;

        const PACKAGE_TYPE* iterA = this->_begin;
        const PACKAGE_TYPE* iterB = tensor._begin;

        PACKAGE_TYPE* iterO = output._begin;

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
    constexpr Tensor<Dimensions...> operator/(const Tensor<Dimensions...>& tensor)
    {
        Tensor<Dimensions...> output;

        const PACKAGE_TYPE* iterA = this->_begin;
        const PACKAGE_TYPE* iterB = tensor._begin;

        PACKAGE_TYPE* iterO = output._begin;

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
    constexpr Tensor<Dimensions...> operator+(Tensor<Dimensions...>&& tensor)
    {
        const PACKAGE_TYPE* iterA = this->_begin;
        PACKAGE_TYPE* iterB = tensor._begin;

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
    constexpr Tensor<Dimensions...> operator-(Tensor<Dimensions...>&& tensor)
    {
        const PACKAGE_TYPE* iterA = this->_begin;
        PACKAGE_TYPE* iterB = tensor._begin;

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
    constexpr Tensor<Dimensions...> operator*(Tensor<Dimensions...>&& tensor)
    {
        const PACKAGE_TYPE* iterA = this->_begin;
        PACKAGE_TYPE* iterB = tensor._begin;

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
    constexpr Tensor<Dimensions...> operator/(Tensor<Dimensions...>&& tensor)
    {
        const PACKAGE_TYPE* iterA = this->_begin;
        PACKAGE_TYPE* iterB = tensor._begin;

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


    void apply_ReLu()
    {
        PACKAGE_TYPE* iter = this->_begin;

        const PACKAGE_TYPE value = _SETZERO();

        while (iter < this->_end)
        {
            *iter = _MAX(*iter, value);

            ++iter;
        }

        if constexpr (_offset)
            _MASKSTORE((float*)iter, remainderMask<_offset>(), _MAX(*iter, value));
    }


    void apply_ReLu_Derivative()
    {
        PACKAGE_TYPE* iter = this->_begin;

        const PACKAGE_TYPE zeros = _SETZERO(),
                            ones = _CASTSI(_SET1_EPI32(-1));

        while (iter < this->_end)
        {
            *iter = _AND(_CMP(*iter, zeros, _CMP_GT_OQ), ones);

            ++iter;
        }

        if constexpr (_offset)
            _MASKSTORE((float*)iter, remainderMask<_offset>(), _AND(_CMP(*iter, zeros, _CMP_GT_OQ), ones));
    }


    constexpr void setMax(float max)
    {
        PACKAGE_TYPE* iter = this->_begin;

        const PACKAGE_TYPE value = _SET1(max);

        while (iter < this->_end)
        {
            *iter = _MAX(*iter, value);

            ++iter;
        }

        if constexpr (_offset)
            _MASKSTORE((float*)iter, remainderMask<_offset>(), _MAX(*iter, value));
    }


    Tensor<Dimensions...> max(const Tensor<Dimensions...> tensor, float max)
    {
        Tensor<Dimensions...> output;

        const PACKAGE_TYPE* iterA = tensor._begin;
        PACKAGE_TYPE* iterO = output._begin;

        const PACKAGE_TYPE value = _SET1(max);

        while (iterO < output._end)
        {
            *iterO = _MAX(*iterA, value);

            ++iterA;
            ++iterO;
        }

        if constexpr (_offset)
            _MASKSTORE((float*)iterO, remainderMask<_offset>(), _MAX(*iterA, value));

        return output;
    }


    // Element-wise addition of tensorC with the element-wise multiplication of tensorA and tensorB
    friend Tensor<Dimensions...> multiply_and_add(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC)
    {
        Tensor<Dimensions...> output;

        const PACKAGE_TYPE* iterA = tensorA._begin;
        const PACKAGE_TYPE* iterB = tensorB._begin;
        const PACKAGE_TYPE* iterC = tensorC._begin;

        PACKAGE_TYPE* iterO = output._begin;

        while (iterO < output._end)
        {
            *iterO = _FMADD(*iterA, *iterB, *iterC);

            ++iterA;
            ++iterB;
            ++iterC;
            ++iterO;
        }

        if constexpr (_offset)
            _MASKSTORE((float*)iterO, remainderMask<_offset>(), _FMADD(*iterA, *iterB, *iterC));

        return output;
    }

    template <size_t... Dimensions>
    friend Tensor<Dimensions...> zeros();

    template <size_t... Dimensions>
    friend Tensor<Dimensions...> rand(float mean, float sigma);

    template <size_t cols, size_t rows, size_t... rest>
    friend Tensor<cols, rows, rest...> transpose(const Tensor<rows, cols, rest...>& tensor);

    // Tensor multiplication between tensorA and tensorB
    template<size_t colsA, size_t rowsA, size_t colsB, size_t... rest>
    friend Tensor<colsB, rowsA, rest...> mul(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsB, colsA, rest...>& tensorB);

    template<size_t colsA, size_t rowsA, size_t colsB, size_t... rest>
    friend Tensor<colsB, rowsA, rest...> mul_transposed(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsA, colsB, rest...>& tensorB);

    template<size_t colsA, size_t rowsA, size_t... rest>
    friend Tensor<rowsA> mul_transposed_scalar(const Tensor<colsA, rowsA>& tensorA, const Tensor<colsA>& tensorB);

    /*template<size_t... dimensions>
    friend Tensor<(1 * ... * dimensions)> flatten(Tensor<dimensions...>& tensor);*/

private:

    PACKAGE_TYPE* _begin = static_cast<PACKAGE_TYPE*>(_mm_malloc(_size * sizeof(float), PACKAGE_ALIGNEMENT));
    const PACKAGE_TYPE* _end = _begin + _numPackages;

    template <size_t... otherDimensions>
    friend class Tensor;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <size_t cols, size_t rows, size_t... rest>
Tensor<cols, rows, rest...> transpose(const Tensor<rows, cols, rest...>& tensor)
{
    Tensor<cols, rows, rest...> output;

    const float* iterA = (float*) tensor._begin;
    float* iterB = (float*) output._begin;

    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            iterB[r * cols + c] = iterA[c * rows + r];
        }
    }

    /*const uint16_t& colsA = matrix.m_fullcols;
    const uint16_t& colsB = output.m_fullcols;

    auto* iterA = tensor._begin;
    auto* iterB = output._begin;

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

        if ((iterB - output.m_begin + 1) % colsB == 0)
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


/*template<size_t... dimensions>
Tensor<(1 * ... * dimensions)> flatten(Tensor<dimensions...>& tensor)
{
    Tensor<(1 * ... * dimensions)> output(std::move(tensor));

    return output;
}*/



// Tensor multiplication between tensorA and tensorB
template<size_t colsA, size_t rowsA, size_t colsB, size_t... rest>
Tensor<colsB, rowsA, rest...> mul(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsB, colsA, rest...>& tensorB)
{
    constexpr uint16_t offcols = colsB % PACKAGE_LENGTH;
    constexpr size_t packagePerColumn = colsB / PACKAGE_LENGTH;

    const auto mask = remainderMask<offcols>();

    Tensor<colsB, rowsA, rest...> output;

    const float* iterA = (float*) tensorA._begin;
    const float* iterB;

    float* iterO = (float*) output._begin;

    #pragma omp parallel do schedule(dynamic)
    do 
    {
        size_t c = 0;

        for (; c < packagePerColumn; c+= PACKAGE_LENGTH)
        {
            PACKAGE_TYPE sum = _SETZERO();
            iterB = (float*) tensorB._begin + c;

            for (size_t i = 0; i < colsA; ++i)
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
            iterB = (float*)tensorB._begin + c;

            for (size_t i = 0; i < colsA; ++i)
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


// Tensor multiplication between tensorA and the transpose of tensorB
template<size_t colsA, size_t rowsA, size_t colsB, size_t... rest>
Tensor<colsB, rowsA, rest...> mul_transposed(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsA, colsB, rest...>& tensorB)
{
    constexpr size_t packagePerColumn = colsA / PACKAGE_LENGTH;
    constexpr auto offcols = colsA % PACKAGE_LENGTH;

    const auto mask = remainderMask<offcols>();

    Tensor<colsB, rowsA, rest...> output;

    float* iterA;
    float* iterB;

    float* iterO = (float*) output._begin;

    #pragma omp parallel for schedule(dynamic)
    for (size_t r = 0; r < rowsA; ++r)
    {
        for (size_t c = 0; c < colsB; ++c)
        {
            PACKAGE_TYPE sum = _SETZERO();

            iterA = (float*) tensorA._begin + colsA * r;
            iterB = (float*) tensorB._begin + colsA * c;

            size_t i = 0;

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



template<size_t colsA, size_t rowsA, size_t... rest>
Tensor<rowsA> mul_transposed_scalar(const Tensor<colsA, rowsA>& tensorA, const Tensor<colsA>& tensorB)
{
    constexpr size_t packagePerColumn = colsA / PACKAGE_LENGTH;
    constexpr auto offcols = colsA % PACKAGE_LENGTH;

    const auto mask = remainderMask<offcols>();

    Tensor<rowsA> output;

    float* iterA;
    float* iterB;

    float* iterO = (float*)output._begin;

    #pragma omp parallel for schedule(dynamic)
    for (size_t r = 0; r < rowsA; ++r)
    {
        PACKAGE_TYPE sum = _SETZERO();

        iterA = (float*)tensorA._begin + colsA * r;
        iterB = (float*)tensorB._begin;

        size_t i = 0;

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



template <size_t... Dimensions>
Tensor<Dimensions...> zeros()
{
    Tensor<Dimensions...> output;
    const uint16_t& offset = output._offset;

    const PACKAGE_TYPE packedValue = _SETZERO();

    PACKAGE_TYPE* iter = output._begin;

    while (iter < output._end)
    {
        *iter = packedValue;

        ++iter;
    }

    if constexpr (offset)
        _MASKSTORE((float*)iter, remainderMask<offset>(), packedValue);

    return output;
}



template <size_t... Dimensions>
Tensor<Dimensions...> rand(float mean, float sigma)
{
    Tensor<Dimensions...> output;
    const uint16_t& offset = output._offset;

    auto seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();// To get differents epochs 
    std::default_random_engine generator(seed);// Create a generator of random numbers

    std::normal_distribution<float> distribution(mean, sigma);

    PACKAGE_TYPE* iter = output._begin;

    while (iter < output._end)
    {
        PACKAGE_TYPE randomValues = _mm256_set_ps(distribution(generator), distribution(generator), distribution(generator), distribution(generator),
            distribution(generator), distribution(generator), distribution(generator), distribution(generator));

        *iter = randomValues;

        ++iter;
    }

    if constexpr (offset)
        _MASKSTORE((float*)iter, remainderMask<offset>(), _mm256_set_ps(distribution(generator), distribution(generator), distribution(generator), distribution(generator),
                                                                        distribution(generator), distribution(generator), distribution(generator), distribution(generator)));

    return output;
}



#ifdef __AVX2__

template<uint16_t offset>
inline static constexpr __m256i remainderMask()
{
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
    // Unaligned load from a constant array
    const int* rsi = &s_remainderLoadMask[offset];
    return _mm256_loadu_si256((const __m256i*)rsi);
}

#endif

