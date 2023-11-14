#pragma once
#include <iostream>
#include "simd_utils.h"
#include "debug.h"

#include <stdio.h>
#include <stdlib.h>


// TODO : remove the useless constexpr ...


template <::std::size_t... Dimensions>
class Tensor 
{
    // Total size of the tensor
    static constexpr ::std::size_t _size = (1 * ... * Dimensions);

    // To make sure the tensor is not empty
    static_assert(_size > 0 && "Tensor size must be > 0");

    // The number of elements which cannot fit in a package
    static constexpr uint16_t _offset = _size % PACKAGE_LENGTH;

    // The number of packages needed to store the tensor including the offset elements
    static constexpr ::std::size_t _numPackages = (_size - _offset) / PACKAGE_LENGTH;

public:


    // ------- TENSORS CONSTRUCTORS -------


    // Default constructor
    constexpr Tensor();


    // Copy constructor
    constexpr Tensor(const Tensor<Dimensions...>& other);


    // Rvalue assignement constructor
    constexpr Tensor(Tensor<Dimensions...>&& other) noexcept;


    // Rvalue assignement constructor for different dimensions Tensor
    template <std::size_t... OtherDimensions>
    constexpr Tensor(Tensor<OtherDimensions...>&& other) noexcept;



    // Fill constructor with a given value
    constexpr Tensor(float value);


    // Fill the tensor from an initializer list
    constexpr Tensor(std::initializer_list<float> values);



    // Fill Tensors with zeros
    template <::std::size_t... Dimensions>
    friend Tensor<Dimensions...> zeros();


    // Fill Tensors with ones
    template <::std::size_t... Dimensions>
    friend Tensor<Dimensions...> ones();


    // Fill Tensors with random values with a distribution based on the given mean and the standard deviation
    template <::std::size_t... Dimensions>
    friend Tensor<Dimensions...> normal(float mean, float std);


    // Destructor
    ~Tensor();


    // ------ TENSORS OPERATORS -------


    // Copy operator with a Tensor of the same dimensions
    constexpr void operator= (const Tensor<Dimensions...>& other);

    // Copy operator with a rvalue Tensor of the same dimensions
    constexpr void operator= (Tensor<Dimensions...>&& other) noexcept;



    // Element-wise addition with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator+(const Tensor<Dimensions...>& tensor);

    // Element-wise substraction with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator-(const Tensor<Dimensions...>& tensor);

    // Element-wise multiplication with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator*(const Tensor<Dimensions...>& tensor);

    // Element-wise division with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator/(const Tensor<Dimensions...>& tensor);



    // Element-wise addition with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator+(Tensor<Dimensions...>&& tensor);

    // Element-wise substraction with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator-(Tensor<Dimensions...>&& tensor);

    // Element-wise multiplication with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator*(Tensor<Dimensions...>&& tensor);

    // Element-wise division with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator/(Tensor<Dimensions...>&& tensor);



    // Element-wise addition of the tensor with another tensor of the same dimensions
    constexpr void operator+=(const Tensor<Dimensions...>& tensor);

    // Element-wise substraction of the tensor with another tensor of the same dimensions
    constexpr void operator-=(const Tensor<Dimensions...>& tensor);

    // Element-wise multiplication of the tensor with another tensor of the same dimensions
    constexpr void operator*=(const Tensor<Dimensions...>& tensor);

    // Element-wise division of the tensor with another tensor of the same dimensions
    constexpr void operator/=(const Tensor<Dimensions...>& tensor);


    // ------ TENSORS BASIC FUNCTIONS -------


    // To print the tensor elements
    template<::std::size_t ...Dimensions>
    friend void print(const Tensor<Dimensions...>& tensor);


    // Reshape the dimensions of the tensor to a compatible one
    template<::std::size_t... newDimensions>
    constexpr Tensor<newDimensions...> reshape();


    // Flatten the tensor to a one dimension tensor
    constexpr Tensor<(1 * ... * Dimensions)> flatten();


    // Compute the absolute value of each element in the tensor
    template<::std::size_t ...Dimensions>
    friend Tensor<Dimensions...> abs(const Tensor<Dimensions...>& tensor);


    // Compute the sum of each values in the tensor
    constexpr float sum();


    // Find the index of the maximum value in the tensor
    constexpr size_t argmax();


    // <!> WARNING : The offset values are compared to zero in the mask | The function horizontal_max8 may not be optimized <!>
    // Find the maximum value in the tensor
    constexpr float max();


    // Compute the mean of the values the tensor
    constexpr float mean();


    // Compute the variance of the values in the tensor based on a given mean
    constexpr float variance(float mean);


    // ------ TENSORS MATRIX OPERATIONS -------


    // Element-wise addition between tensorC and the element-wise multiplication of tensorA and tensorB
    // @return tensorA * tensorB + tensorC
    friend Tensor<Dimensions...> multiply_and_add(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC);


    // Element-wise substraction between tensorC and the element-wise multiplication of tensorA and tensorB
    // @return tensorA * tensorB - tensorC
    friend Tensor<Dimensions...> multiply_and_sub(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC);


    // Transpose the tensor
    template <::std::size_t cols, ::std::size_t rows, ::std::size_t... rest>
    friend Tensor<cols, rows, rest...> transpose(const Tensor<rows, cols, rest...>& tensor);


    // Matrix multiplication between tensorA and tensorB
    template<::std::size_t colsA, ::std::size_t rowsA, ::std::size_t colsB, ::std::size_t... rest>
    friend Tensor<colsB, rowsA, rest...> mul(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsB, colsA, rest...>& tensorB);


    // <!> WARNING : All the tensors dimensions are not taken into account <!> 
    // Matrix multiplication between tensorA and the transpose of tensorB
    template<::std::size_t colsA, ::std::size_t rowsA, ::std::size_t colsB, ::std::size_t... rest>
    friend Tensor<colsB, rowsA, rest...> mul_transposed(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsA, colsB, rest...>& tensorB);


    // Matrix multiplication between tensorA and the transpose of tensorB as a scalar
    template<::std::size_t colsA, ::std::size_t rowsA>
    friend Tensor<rowsA> mul_transposed_scalar(const Tensor<colsA, rowsA>& tensorA, const Tensor<colsA>& tensorB);


    // Matrix multiplication between the transpose of tensorA and tensorB both as a scalar
    template<::std::size_t colsA, ::std::size_t colsB>
    friend Tensor<colsB, colsA> mul_transposed_scalar(const Tensor<colsA>& tensorA, const Tensor<colsB>& tensorB);


    // ------ TENSORS LAYERS FUNCTIONS -------


    // Apply the sigmoid function on the tensor values
    void apply_sigmoid();


    // Apply the ReLu function on the tensor values
    void apply_ReLu();

    // Apply the derivative of the sigmoid function on the tensor values
    void apply_ReLu_derivative();


    // Apply a normal distribution on the tensor values
    void apply_normalization();

    // Normalize the tensor based on a given mean and variance then shift and scale the values
    void norm_shift_and_scale(float mean, float variance, float shift, float scale);


private:
    
    // Pointer to the tensor values
    PACKAGE_TYPE* _values;

    // To get access to private members of Tensor with different dimensions
    template <::std::size_t... otherDimensions>
    friend class Tensor;
};

#include "Tensor.cpp"
#include "network_utils.cpp"
