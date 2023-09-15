#include "MatrixF.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <immintrin.h>
#include "debug.h"
#include <sstream>


float sum8(const __m256& x);

#define packageSize 8

MatrixF::MatrixF() :
    m_rows(0), m_cols(0), m_begin(nullptr), m_end(nullptr), m_length(0), m_packageEnd(nullptr), m_packageBack(0), m_packageCount(0)
{
}

MatrixF::MatrixF(const uint16_t rows, const uint16_t cols) :
    m_rows(rows), m_cols(cols), m_length(rows* cols), m_begin(static_cast<float*>(_mm_malloc(m_length * sizeof(float), 16))), m_end(m_begin + m_length), m_packageCount(m_length / packageSize), m_packageBack(m_packageCount* packageSize), m_packageEnd(m_begin + m_packageCount * packageSize)
{
}

MatrixF::MatrixF(const uint16_t rows, const uint16_t cols, float value) :
    m_rows(rows), m_cols(cols), m_length(rows* cols), m_begin(static_cast<float*>(_mm_malloc(m_length * sizeof(float), 16))), m_end(m_begin + m_length), m_packageCount(m_length / packageSize), m_packageBack(m_packageCount* packageSize), m_packageEnd(m_begin + m_packageCount * packageSize)
{
    __m256 val = _mm256_set1_ps(value);

    auto it = m_begin;

    #pragma omp parallel for schedule(dynamic)
    for (; it < m_packageEnd; it += packageSize)
    {
        _mm256_store_ps(it, val);
    }

    #pragma omp parallel
    for (; it < m_end; ++it)
    {
        *it = value;
    }
}

MatrixF::MatrixF(const uint16_t rows, const uint16_t cols, float* values) :
    m_rows(rows), m_cols(cols), m_length(rows* cols), m_begin(values), m_end(m_begin + m_length), m_packageCount(m_length / packageSize), m_packageBack(m_packageCount* packageSize), m_packageEnd(m_begin + m_packageCount * packageSize)
{
}


MatrixF::MatrixF(const MatrixF& other) :
    m_rows(other.m_rows), m_cols(other.m_cols), m_length(other.m_length), m_begin(static_cast<float*>(_mm_malloc(m_length * sizeof(float), 16))), m_end(m_begin + m_length), m_packageCount(m_length / packageSize), m_packageBack(m_packageCount* packageSize), m_packageEnd(m_begin + m_packageCount * packageSize)
{
    std::memcpy(m_begin, other.m_begin, m_length * sizeof(float));
}


MatrixF::~MatrixF()
{
    if (m_begin)
        _mm_free(m_begin);
}


uint16_t MatrixF::rows() const
{
    return m_rows;
}


uint16_t MatrixF::cols() const
{
    return m_cols;
}


MatrixF MatrixF::operator+(const float value) const
{
    MatrixF output(this->m_rows, this->m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(this->m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(output.m_begin);

    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesB[i] = _mm256_add_ps(packedValuesA[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        output.m_begin[i] = this->m_begin[i] + value;
    }

    return output;
}


MatrixF MatrixF::operator-(const float value) const
{
    MatrixF output(this->m_rows, this->m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(this->m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(output.m_begin);

    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesB[i] = _mm256_sub_ps(packedValuesA[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        output.m_begin[i] = this->m_begin[i] - value;
    }

    return output;
}


MatrixF MatrixF::operator*(const float value) const
{
    MatrixF output(this->m_rows, this->m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(this->m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(output.m_begin);

    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesB[i] = _mm256_mul_ps(packedValuesA[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        output.m_begin[i] = this->m_begin[i] * value;
    }

    return output;
}


MatrixF MatrixF::operator/(const float value) const
{
    MatrixF output(this->m_rows, this->m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(this->m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(output.m_begin);

    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesB[i] = _mm256_div_ps(packedValuesA[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        output.m_begin[i] = this->m_begin[i] / value;
    }

    return output;
}


MatrixF MatrixF::operator+(const MatrixF& other) const
{
    debug_assert(this->m_rows == other.m_rows && this->m_cols == other.m_cols && "Error in MatrixF operator+: the given matrices are not of the same size.");


    MatrixF output(this->m_rows, this->m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(this->m_begin);
    const __m256* packedValuesB = reinterpret_cast<const __m256*>(other.m_begin);

    __m256* packedValuesC = reinterpret_cast<__m256*>(output.m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesC[i] = _mm256_add_ps(packedValuesA[i], (const __m256)packedValuesB[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        output.m_begin[i] = this->m_begin[i] + other.m_begin[i];
    }

    return output;
}



MatrixF MatrixF::operator-(const MatrixF& other) const
{
    debug_assert(this->m_rows == other.m_rows && this->m_cols == other.m_cols && "Error in MatrixF operator- the given matrices are not of the same size.");

    MatrixF output(this->m_rows, this->m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(this->m_begin);
    const __m256* packedValuesB = reinterpret_cast<const __m256*>(other.m_begin);

    __m256* packedValuesC = reinterpret_cast<__m256*>(output.m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesC[i] = _mm256_sub_ps(packedValuesA[i], (const __m256)packedValuesB[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (i *= 8; i < this->m_length; ++i)
    {
        output.m_begin[i] = this->m_begin[i] - other.m_begin[i];
    }

    return output;
}

MatrixF MatrixF::operator*(const MatrixF& other) const
{
    debug_assert(this->m_rows == other.m_rows && this->m_cols == other.m_cols && "Error in MatrixF operator* the given matrices are not of the same size.");

    MatrixF output(this->m_rows, this->m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(this->m_begin);
    const __m256* packedValuesB = reinterpret_cast<const __m256*>(other.m_begin);

    __m256* packedValuesC = reinterpret_cast<__m256*>(output.m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesC[i] = _mm256_mul_ps(packedValuesA[i], (const __m256)packedValuesB[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        output.m_begin[i] = this->m_begin[i] * other.m_begin[i];
    }

    return output;
}


MatrixF MatrixF::operator/(const MatrixF& other) const
{
    debug_assert(this->m_rows == other.m_rows && this->m_cols == other.m_cols && "Error in MatrixF operator/ the given matrices are not of the same size.");

    MatrixF output(this->m_rows, this->m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(this->m_begin);
    const __m256* packedValuesB = reinterpret_cast<const __m256*>(other.m_begin);

    __m256* packedValuesC = reinterpret_cast<__m256*>(output.m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesC[i] = _mm256_div_ps(packedValuesA[i], (const __m256)packedValuesB[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        output.m_begin[i] = this->m_begin[i] / other.m_begin[i];
    }

    return output;
}


MatrixF& MatrixF::operator=(const MatrixF& other)
{
    if (this == &other) {
        return *this;  // Self-assignment, no action needed
    }

    if (m_length != other.m_length)
    {
        // Delete the existing values
        if (m_begin)
            _mm_free(m_begin);

        m_length = other.m_length;
        m_packageCount = m_length / packageSize;
        m_packageBack = m_packageCount * packageSize;

        // Allocate new memory for values with proper alignment
        m_begin = static_cast<float*>(_mm_malloc(m_length * sizeof(float), 16));
        m_end = m_begin + m_length;
    }

    // Copy the dimensions
    m_rows = other.m_rows;
    m_cols = other.m_cols;

    std::memcpy(m_begin, other.m_begin, m_length * sizeof(float));

    return *this;
}


void MatrixF::operator+=(const float value)
{
    __m256* packedValues = reinterpret_cast<__m256*>(this->m_begin);
    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValues[i] = _mm256_add_ps(packedValues[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        this->m_begin[i] += value;
    }
}

void MatrixF::operator-=(const float value)
{
    __m256* packedValues = reinterpret_cast<__m256*>(this->m_begin);
    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValues[i] = _mm256_sub_ps(packedValues[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        this->m_begin[i] -= value;
    }
}


void MatrixF::operator*=(const float value)
{
    __m256* packedValues = reinterpret_cast<__m256*>(this->m_begin);
    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValues[i] = _mm256_mul_ps(packedValues[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        this->m_begin[i] *= value;
    }
}


void MatrixF::operator/=(const float value)
{
    __m256* packedValues = reinterpret_cast<__m256*>(this->m_begin);
    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValues[i] = _mm256_div_ps(packedValues[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        this->m_begin[i] /= value;
    }
}


void MatrixF::operator+=(const MatrixF& other)
{
    debug_assert(this->m_rows == other.m_rows && this->m_cols == other.m_cols && "Error in MatrixF operator+= the given matrices are not of the same size.");

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(other.m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(this->m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesB[i] = _mm256_add_ps(packedValuesB[i], (const __m256)packedValuesA[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        this->m_begin[i] += other.m_begin[i];
    }

}

void MatrixF::operator-=(const MatrixF& other)
{
    debug_assert(this->m_rows == other.m_rows && this->m_cols == other.m_cols && "Error in MatrixF operator-= the given matrices are not of the same size.");

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(other.m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(this->m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesB[i] = _mm256_sub_ps(packedValuesB[i], (const __m256)packedValuesA[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        this->m_begin[i] -= other.m_begin[i];
    }
}

void MatrixF::operator*=(const MatrixF& other)
{
    debug_assert(this->m_rows == other.m_rows && this->m_cols == other.m_cols && "Error in MatrixF operator*= the given matrices are not of the same size.");

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(other.m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(this->m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesB[i] = _mm256_mul_ps(packedValuesB[i], (const __m256)packedValuesA[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        this->m_begin[i] *= other.m_begin[i];
    }
}

void MatrixF::operator/=(const MatrixF& other)
{
    debug_assert(this->m_rows == other.m_rows && this->m_cols == other.m_cols && "Error in MatrixF operator/= the given matrices are not of the same size.");

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(other.m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(this->m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < m_packageCount; ++i)
    {
        packedValuesB[i] = _mm256_div_ps(packedValuesB[i], (const __m256)packedValuesA[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (i = m_packageBack; i < this->m_length; ++i)
    {
        this->m_begin[i] /= other.m_begin[i];
    }
}


float& MatrixF::operator()(const uint16_t row, const uint16_t col)
{
    debug_assert(row < m_rows&& col < m_cols && "Error in MatrixF operator() the given position is out of the matrix.");
    return this->m_begin[row * this->m_cols + col];
}


float MatrixF::operator()(const uint16_t row, const uint16_t col) const
{
    debug_assert(row < m_rows&& col < m_cols && "Error in MatrixF operator() the given position is out of the matrix.");
    return this->m_begin[row * this->m_cols + col];
}


void add(const MatrixF& a, const float value, MatrixF& output)
{
    output = MatrixF(a.m_rows, a.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(a.m_begin);
    __m256* packedValues = reinterpret_cast<__m256*>(output.m_begin);
    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < output.m_length / packageSize; ++i)
    {
        packedValues[i] = _mm256_add_ps(packedValuesA[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = output.m_packageBack; i < output.m_length; ++i)
    {
        output.m_begin[i] = a.m_begin[i] + value;
    }
}


void sub(const MatrixF& a, const float value, MatrixF& output)
{
    output = MatrixF(a.m_rows, a.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(a.m_begin);
    __m256* packedValues = reinterpret_cast<__m256*>(output.m_begin);
    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < output.m_length / packageSize; ++i)
    {
        packedValues[i] = _mm256_sub_ps(packedValuesA[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = output.m_packageBack; i < output.m_length; ++i)
    {
        output.m_begin[i] = a.m_begin[i] - value;
    }
}


void mul(const MatrixF& a, const float value, MatrixF& output)
{
    output = MatrixF(a.m_rows, a.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(a.m_begin);
    __m256* packedValues = reinterpret_cast<__m256*>(output.m_begin);
    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < output.m_length / packageSize; ++i)
    {
        packedValues[i] = _mm256_mul_ps(packedValuesA[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = output.m_packageBack; i < output.m_length; ++i)
    {
        output.m_begin[i] = a.m_begin[i] * value;
    }
}


void div(const MatrixF& a, const float value, MatrixF& output)
{
    output = MatrixF(a.m_rows, a.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(a.m_begin);
    __m256* packedValues = reinterpret_cast<__m256*>(output.m_begin);
    const __m256 val = _mm256_set1_ps(value);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < output.m_length / packageSize; ++i)
    {
        packedValues[i] = _mm256_div_ps(packedValuesA[i], val);
    }

    // Process remaining elements (not multiple of 8)
    for (i = output.m_packageBack; i < output.m_length; ++i)
    {
        output.m_begin[i] = a.m_begin[i] / value;
    }
}


void add(const MatrixF& a, const MatrixF& b, MatrixF& output)
{
    debug_assert(a.m_rows == b.m_rows && a.m_cols == b.m_cols && "Error in MatrixF add : the given matrices are not of the same size.");

    MatrixF result(a.m_rows, a.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(a.m_begin);
    const __m256* packedValuesB = reinterpret_cast<const __m256*>(b.m_begin);

    __m256* packedValuesC = reinterpret_cast<__m256*>(result.m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < result.m_packageCount; ++i)
    {
        packedValuesC[i] = _mm256_add_ps(packedValuesA[i], (const __m256)packedValuesB[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (i = result.m_packageBack; i < result.m_length; ++i)
    {
        result.m_begin[i] = a.m_begin[i] + b.m_begin[i];
    }

    move(result, output);
}


void sub(const MatrixF& a, const MatrixF& b, MatrixF& output)
{
    debug_assert(a.m_rows == b.m_rows && a.m_cols == b.m_cols && "Error in MatrixF sub : the given matrices are not of the same size.");

    MatrixF result(a.m_rows, a.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(a.m_begin);
    const __m256* packedValuesB = reinterpret_cast<const __m256*>(b.m_begin);

    __m256* packedValuesC = reinterpret_cast<__m256*>(result.m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < a.m_packageCount; ++i)
    {
        packedValuesC[i] = _mm256_sub_ps(packedValuesA[i], packedValuesB[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (uint32_t i = result.m_packageBack; i < result.m_length; ++i)
    {
        result.m_begin[i] = a.m_begin[i] - b.m_begin[i];
    }

    move(result, output);
}


void mul(const MatrixF& a, const MatrixF& b, MatrixF& output)
{
    debug_assert(a.m_rows == b.m_rows && a.m_cols == b.m_cols && "Error in MatrixF sub : the given matrices are not of the same size.");

    MatrixF result(a.m_rows, a.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(a.m_begin);
    const __m256* packedValuesB = reinterpret_cast<const __m256*>(b.m_begin);

    __m256* packedValuesC = reinterpret_cast<__m256*>(result.m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < a.m_packageCount; ++i)
    {
        packedValuesC[i] = _mm256_mul_ps(packedValuesA[i], packedValuesB[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (uint32_t i = result.m_packageBack; i < result.m_length; ++i)
    {
        result.m_begin[i] = a.m_begin[i] * b.m_begin[i];
    }

    move(result, output);
}


void div(const MatrixF& a, const MatrixF& b, MatrixF& output)
{
    debug_assert(a.m_rows == b.m_rows && a.m_cols == b.m_cols && "Error in MatrixF sub : the given matrices are not of the same size.");

    MatrixF result(a.m_rows, a.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(a.m_begin);
    const __m256* packedValuesB = reinterpret_cast<const __m256*>(b.m_begin);

    __m256* packedValuesC = reinterpret_cast<__m256*>(result.m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < a.m_packageCount; ++i)
    {
        packedValuesC[i] = _mm256_div_ps(packedValuesA[i], packedValuesB[i]);
    }

    // Process remaining elements (not multiple of 8)
    for (uint32_t i = result.m_packageBack; i < result.m_length; ++i)
    {
        result.m_begin[i] = a.m_begin[i] / b.m_begin[i];
    }

    move(result, output);
}


float sum(const MatrixF& matrix)
{
    uint32_t i;

    __m256 sum = _mm256_setzero_ps();

    const __m256* packedValues = reinterpret_cast<const __m256*>(matrix.m_begin);

    #pragma omp parallel for schedule(dynamic) reduction(+: sum)
    // Process the elements in packages using SIMD instructions
    for (i = 0; i < matrix.m_length / packageSize; ++i)
    {
        sum = _mm256_add_ps(packedValues[i], sum);
    }

    // Process the remaining elements individually
    for (i = matrix.m_packageBack; i < matrix.m_length; ++i)
    {
        sum.m256_f32[0] += matrix.m_begin[i];
    }

    return sum8(sum);
}


float max(const MatrixF& matrix)
{
    uint32_t i;

    const __m256* packedValues = reinterpret_cast<const __m256*>(matrix.m_begin);

    __m256 max = packedValues[0];

    #pragma omp parallel for schedule(dynamic) reduction(max: max)
    // Process the elements in packages using SIMD instructions
    for (i = 1; i < matrix.m_length / packageSize; ++i)
    {
        max = _mm256_max_ps(packedValues[i], max);
    }

    i = matrix.m_packageBack;
    float output = matrix.m_begin[i];
    ++i;

    // Find the maximum value in the remaining elements individually
    for (; i < matrix.m_length; ++i)
    {
        const float& value = matrix.m_begin[i];

        if (output < value)
            output = value;
    }

    // Find the maximum value from the SIMD vector
    for (i = 0; i < packageSize; ++i)
    {
        const float& value = max.m256_f32[i];

        if (output < value)
            output = value;
    }

    return output;
}

uint32_t argmax(const MatrixF& matrix)
{
    uint32_t id_max = 0;

    #pragma omp parallel for schedule(dynamic) reduction(max: id_max)
    for (uint32_t i = 1; i < matrix.m_length; ++i)
    {
        if (matrix.m_begin[id_max] < matrix.m_begin[i])
        {
            id_max = i;
        }
    }
    return id_max;
}


MatrixF abs(const MatrixF& matrix)
{
    MatrixF output(matrix.rows(), matrix.cols());

    uint32_t i;

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(matrix.m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(output.m_begin);

    const __m256 set0 = _mm256_set1_ps(-0.0f);

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < matrix.m_length / packageSize; ++i)
    {
        packedValuesB[i] = _mm256_andnot_ps(set0, (const __m256)packedValuesA[i]);
    }

    // Process multiple of 8 at once
    for (i = matrix.m_packageBack; i < matrix.m_length; ++i)
    {
        output.m_begin[i] = std::abs(matrix.m_begin[i]);
    }

    return output;
}


MatrixF exp(const MatrixF& matrix)
{
    MatrixF output(matrix.m_rows, matrix.m_cols);

    uint32_t i;

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(matrix.m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(output.m_begin);

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < output.m_length / packageSize; ++i)
    {
        packedValuesB[i] = _mm256_exp_ps(packedValuesA[i]);
    }

    // Process multiple of 8 at once
    for (i = matrix.m_packageBack; i < output.m_length; ++i)
    {
        output.m_begin[i] = exp(matrix.m_begin[i]);
    }

    return output;
}


void print(const MatrixF& matrix, float decimals)
{
    const float coef = powf(10.f, decimals);

    float* iter = matrix.m_begin;

    #pragma omp parallel for schedule(dynamic)
    for (uint16_t r = 0; r < matrix.m_rows; ++r)
    {
        std::stringstream ss;

        for (; iter < matrix.m_begin + matrix.m_cols * (r + 1); ++iter)
        {
            ss << roundf(*iter * coef) / coef << " ";
        }

        #pragma omp critical
        std::cout << ss.str() << "\n";
    }
}


MatrixF relu(const MatrixF& matrix)
{
    MatrixF output(matrix.m_rows, matrix.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(matrix.m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(output.m_begin);

    __m256 set0 = _mm256_setzero_ps();

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < matrix.m_length / packageSize; ++i)
    {
        packedValuesB[i] = _mm256_max_ps(packedValuesA[i], set0);
    }

    // Process remaining elements (not multiple of 8)
    for (i = matrix.m_packageBack; i < matrix.m_length; ++i)
    {
        const float& value = matrix.m_begin[i];

        output.m_begin[i] = value > 0.f ? value : 0.f;
    }

    return output;
}


MatrixF relu_derivative(const MatrixF& matrix)
{
    MatrixF output(matrix.m_rows, matrix.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(matrix.m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(output.m_begin);

    const __m256 set1 = _mm256_set1_ps(1),
        set0 = _mm256_setzero_ps();

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < matrix.m_length / packageSize; ++i)
    {

        packedValuesB[i] = _mm256_and_ps(_mm256_cmp_ps((const __m256)packedValuesA[i], set0, _CMP_GT_OQ), set1);
    }

    // Process remaining elements (not multiple of 8)
    for (i = matrix.m_packageBack; i < matrix.m_length; ++i)
    {
        output.m_begin[i] = matrix.m_begin[i] > 0.f ? 1.f : 0.f;
    }

    return output;
}


MatrixF leaky_relu(const MatrixF& matrix, const float slope)
{
    MatrixF output(matrix.m_rows, matrix.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(matrix.m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(output.m_begin);

    const __m256 set0 = _mm256_set1_ps(slope);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < matrix.m_length / packageSize; ++i)
    {
        const __m256& packageA = packedValuesA[i];

        packedValuesB[i] = _mm256_max_ps(packageA, _mm256_mul_ps(packageA, set0));
    }

    // Process remaining elements (not multiple of 8)
    for (i = matrix.m_packageBack; i < matrix.m_length; ++i)
    {
        const float& value = matrix.m_begin[i];

        output.m_begin[i] = value >= 0.f ? value : slope * value;
    }

    return output;
}


MatrixF leaky_relu_derivative(const MatrixF& matrix, const float slope)
{
    MatrixF output(matrix.m_rows, matrix.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(matrix.m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(output.m_begin);

    const __m256 set0 = _mm256_setzero_ps(),
        set1 = _mm256_set1_ps(1),
        set2 = _mm256_set1_ps(slope);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once
    for (i = 0; i < matrix.m_length / packageSize; ++i)
    {
        packedValuesB[i] = _mm256_blendv_ps(set2, set1, _mm256_cmp_ps((__m256)packedValuesA[i], set0, _CMP_GT_OQ));
    }

    // Process remaining elements (not multiple of 8)
    for (i = matrix.m_packageBack; i < matrix.m_length; ++i)
    {
        output.m_begin[i] = matrix.m_begin[i] > 0.f ? 1.f : slope;
    }

    return output;
}


MatrixF sigmoid(const MatrixF& matrix)
{
    MatrixF output(matrix.m_rows, matrix.m_cols);

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(matrix.m_begin);
    __m256* packedValuesB = reinterpret_cast<__m256*>(output.m_begin);

    const __m256 set1 = _mm256_set1_ps(1),
        set0 = _mm256_setzero_ps();

    uint32_t i;

    #pragma omp parallel for schedule(dynamic)
    // Process multiple of 8 at once using SIMD instructions
    for (i = 0; i < matrix.m_length / packageSize; ++i)
    {
        const __m256 expValues = _mm256_exp_ps(_mm256_sub_ps(set0, (__m256)packedValuesA[i]));  // Compute exponent

        packedValuesB[i] = _mm256_div_ps(set1, _mm256_add_ps(set1, expValues));  // Compute sigmoid
    }

    // Process remaining elements (not multiple of 8)
    for (i = matrix.m_packageBack; i < matrix.m_length; ++i)
    {
        output.m_begin[i] = 1.0f / (1.0f + std::exp(-matrix.m_begin[i]));  // Compute sigmoid
    }

    return output;
}


MatrixF softmax(const MatrixF& matrix)
{
    const MatrixF output = exp(matrix - max(matrix));

    return output / sum(output);
}


MatrixF softmax_derivative(const MatrixF& matrix)
{
    return matrix - scaled_sum(matrix, matrix);
}


MatrixF transpose(const MatrixF& matrix)
{
    MatrixF output(matrix.m_cols, matrix.m_rows);

    uint16_t r, c;

    float* iter = matrix.m_begin;

    #pragma omp parallel for schedule(dynamic)
    for (r = 0; r < matrix.m_rows / 4; r += 4)
    {
        for (c = 0; c < matrix.m_cols / 4; c += 4)
        {
            float* outputInit = output.m_begin + c * output.m_cols + r;

            __m128 row1 = _mm_load_ps(iter + 0 * matrix.m_cols);
            __m128 row2 = _mm_load_ps(iter + 1 * matrix.m_cols);
            __m128 row3 = _mm_load_ps(iter + 2 * matrix.m_cols);
            __m128 row4 = _mm_load_ps(iter + 3 * matrix.m_cols);

            _MM_TRANSPOSE4_PS(row1, row2, row3, row4);

            _mm_store_ps(outputInit + 0 * output.m_cols, row1);
            _mm_store_ps(outputInit + 1 * output.m_cols, row2);
            _mm_store_ps(outputInit + 2 * output.m_cols, row3);
            _mm_store_ps(outputInit + 3 * output.m_cols, row4);

            iter += 4;
        }
        for (; c < matrix.m_cols; ++c)
        {
            float* outputInit = output.m_begin + c * output.m_cols + r;

            __m128 row = _mm_set_ps(*(iter + 3 * matrix.m_cols), *(iter + 2 * matrix.m_cols),
                *(iter + 1 * matrix.m_cols), *(iter + 0 * matrix.m_cols));

            _mm_store_ps(outputInit, row);

            ++iter;
        }

        iter += 3 * matrix.m_cols;
    }
    for (; r < matrix.m_rows; ++r)
    {
        for (c = 0; c < matrix.m_cols; ++c)
        {
            *(output.m_begin + c * output.m_cols + r) = *iter;

            ++iter;
        }
    }

    return output;
}


void swap(MatrixF& a, MatrixF& b)
{
    std::swap(a.m_rows, b.m_rows);
    std::swap(a.m_cols, b.m_cols);
    std::swap(a.m_length, b.m_length);
    std::swap(a.m_begin, b.m_begin);
    std::swap(a.m_end, b.m_end);
}


void move(MatrixF& a, MatrixF& b)
{
    b.m_rows = a.m_rows;
    b.m_cols = a.m_cols;
    b.m_length = a.m_length;
    b.m_begin = a.m_begin;
    b.m_end = a.m_end;

    a.m_begin = nullptr;
}


MatrixF dot(const MatrixF& a, const MatrixF& b)
{
    debug_assert(a.m_cols == b.m_rows && "Error in MatrixF dot the two given matrices don't have the relation matrixA cols equal to matrixB rows");

    MatrixF output(a.m_rows, b.m_cols);

    const uint16_t packageBack1 = (output.m_cols / packageSize) * packageSize;
    const uint16_t packageBack2 = (a.m_cols / packageSize) * packageSize;

    float* iter = output.m_begin;

    __m256 sum;

    uint16_t r, c, i;

    float* iterA;
    float* iterB;

    #pragma omp parallel for schedule(dynamic)
    for (r = 0; r < output.m_rows; ++r)
    {
        iterA = a.m_begin + r * a.m_cols;

        for (c = 0; c < packageBack1; c += packageSize)
        {
            sum = _mm256_setzero_ps();

            iterB = b.m_begin + c;

            for (i = 0; i < packageBack2; i += packageSize)
            {
                sum = _mm256_fmadd_ps(_mm256_load_ps(iterB += b.m_cols), _mm256_set1_ps(*iterA++), _mm256_fmadd_ps(_mm256_load_ps(iterB += b.m_cols), _mm256_set1_ps(*iterA++),
                      _mm256_fmadd_ps(_mm256_load_ps(iterB += b.m_cols), _mm256_set1_ps(*iterA++), _mm256_fmadd_ps(_mm256_load_ps(iterB += b.m_cols), _mm256_set1_ps(*iterA++),
                      _mm256_fmadd_ps(_mm256_load_ps(iterB += b.m_cols), _mm256_set1_ps(*iterA++), _mm256_fmadd_ps(_mm256_load_ps(iterB += b.m_cols), _mm256_set1_ps(*iterA++),
                      _mm256_fmadd_ps(_mm256_load_ps(iterB += b.m_cols), _mm256_set1_ps(*iterA++), _mm256_fmadd_ps(_mm256_load_ps(iterB), _mm256_set1_ps(*iterA++),
                      sum))))))));

                iterB += b.m_cols;
            }

            for (; i < a.m_cols; ++i)
            {
                sum = _mm256_fmadd_ps(_mm256_set1_ps(*iterA++), _mm256_load_ps(iterB), sum);
                iterB += b.m_cols;
            }

            _mm256_store_ps(iter, sum);

            iter += packageSize;
            iterA -= a.m_cols;
        }

        for (; c < output.m_cols; ++c)
        {
            sum = _mm256_setzero_ps();

            float* iterA = a.m_begin + r * a.m_cols;
            float* iterB = b.m_begin + c;

            for (i = 0; i < packageBack2; i += packageSize)
            {
                const __m256 packageB = _mm256_set_ps(*(iterB + 7 * b.m_cols), *(iterB + 6 * b.m_cols), *(iterB + 5 * b.m_cols), *(iterB + 4 * b.m_cols),
                                                      *(iterB + 3 * b.m_cols), *(iterB + 2 * b.m_cols), *(iterB + b.m_cols), *iterB);

                sum = _mm256_fmadd_ps(_mm256_load_ps(iterA), packageB, sum);

                iterA += packageSize;
                iterB += packageSize * b.m_cols;
            }

            for (; i < a.m_cols; ++i)
            {
                sum.m256_f32[0] += (*iterA++) * (*iterB);

                iterB += b.m_cols;
            }

            (*iter++) = sum8(sum);
        }
    }

    return output;
}


float scalar(const MatrixF& a, const MatrixF& b)
{
    debug_assert(a.m_rows == b.m_rows && a.m_cols == b.m_cols && "Error in MatrixF scalar method the given inputs has not the same size");


    __m256 sum = _mm256_setzero_ps();
    const __m256* packedValuesA = reinterpret_cast<const __m256*>(a.m_begin);
    const __m256* packedValuesB = reinterpret_cast<const __m256*>(b.m_begin);

    uint32_t i;

    #pragma omp parallel for schedule(dynamic) reduction(+:sum)
    // Process multiple of 8 at once using SIMD instructions
    for (i = 0; i < a.m_length / packageSize; ++i)
    {
        sum = _mm256_fmadd_ps(packedValuesA[i], (const __m256)packedValuesB[i], sum);
    }

    // Process remaining elements (not multiple of 8)
    for (i = a.m_packageBack; i < a.m_length; ++i)
    {
        sum.m256_f32[0] += a.m_begin[i] * b.m_begin[i];
    }

    return sum8(sum);
}


MatrixF scaled_sum(const MatrixF& a, const MatrixF& b)
{
    debug_assert(a.m_rows == b.m_rows && a.m_cols == b.m_cols && "Error in MatrixF scalar method the given inputs has not the same size");

    MatrixF output(a.m_rows, a.m_cols, 0.f);

    const uint32_t packageEnd = b.m_length - packageSize + 1;

    const __m256* packedValuesA = reinterpret_cast<const __m256*>(a.m_begin);
    const __m256* packedValuesB = reinterpret_cast<const __m256*>(b.m_begin);
    __m256* packedValuesC = reinterpret_cast<__m256*>(output.m_begin);

    uint32_t i, j;

    #pragma omp parallel for schedule(dynamic) private(j)
    // Process multiple of 8 at once using SIMD instructions
    for (i = 0; i < a.m_length / packageSize; ++i)
    {
        const __m256& packageA = packedValuesA[i];
        __m256& packageC = packedValuesC[i];

        for (j = 0; j < packageEnd; ++j)
        {
            packageC = _mm256_fmadd_ps(packageA, _mm256_load_ps(b.m_begin + j), (__m256)packageC);
        }

        for (; j < b.m_length; ++j)
        {
            const __m256 packageB = _mm256_set_ps(b.m_begin[(j + 7) % b.m_length], b.m_begin[(j + 6) % b.m_length], b.m_begin[(j + 5) % b.m_length], b.m_begin[(j + 4) % b.m_length],
                                                  b.m_begin[(j + 3) % b.m_length], b.m_begin[(j + 2) % b.m_length], b.m_begin[(j + 1) % b.m_length], b.m_begin[j % b.m_length]);

            packageC = _mm256_fmadd_ps(packageA, packageB, (__m256)packageC);
        }
    }

    #pragma omp parallel for schedule(dynamic) private(j)
    // Process remaining elements (not multiple of 8)
    for (i = a.m_packageBack; i < a.m_length; ++i)
    {
        float& valueC = output.m_begin[i];
        const float& valueA = a.m_begin[i];

        for (j = 0; j < b.m_length; ++j)
        {
            const float& valueB = b.m_begin[j];

            valueC += valueA * valueB;
        }

    }

    return output;
}


void fill(MatrixF& a, float value)
{
    float* p = a.m_begin;
    const __m256 val = _mm256_set1_ps(value);

    #pragma omp parallel for schedule(static)
    // Process multiple of 8 at once
    for (; p < a.m_packageEnd;)
    {
        _mm256_store_ps(p, val);

        p += packageSize;
    }

    #pragma omp parallel for schedule(dynamic, 1)
    // Process remaining elements (not multiple of 8)
    for (; p < a.m_end;)
    {
        (*p++) = value;
    }
}


MatrixF rand(const uint16_t rows, const uint16_t cols, const float mean, const float sigma)
{
    MatrixF output(rows, cols);

    float* p = output.m_begin;
    __m256 randomValues;

    auto seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();// To get differents epochs 
    std::default_random_engine generator(seed);// Create a generator of random numbers

    std::normal_distribution<float> distribution(mean, sigma);

    #pragma omp parallel for schedule(static)
    // Process multiple of 8 at once
    for (; p < output.m_packageEnd; p += packageSize)
    {
        randomValues = _mm256_set_ps(distribution(generator), distribution(generator), distribution(generator), distribution(generator),
                                     distribution(generator), distribution(generator), distribution(generator), distribution(generator));

        _mm256_store_ps(p, randomValues);
    }

    #pragma omp parallel for schedule(dynamic, 1)
    // Process remaining elements (not multiple of 8)
    for (; p < output.m_end; ++p)
    {
        (*p) = distribution(generator);
    }

    return output;
}


float sum8(const __m256& x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);

    return _mm_cvtss_f32(sum);
}