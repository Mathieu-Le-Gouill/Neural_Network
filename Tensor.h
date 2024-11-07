#pragma once
#include <iostream>
#include "simd_utils.h"
#include "debug.h"

#include <stdio.h>
#include "network_parameters.h"
#include <array>
#include <stdlib.h>
#include <vector>


// TEST if it is possible to make a lot a function other dimensions with the restriction of the size
// Create Reshape function with another function name.
// Reduce  input data unnecessary dimensionality

// TODO : remove the useless constexpr ...
// For the parallelization, there should be two similar privates inline functions for with and without parallelization
// Then the public function should call on of them based on a constexpr condition to check if it's worth it.

template <std::size_t... Dimensions>
class Tensor;

template<std::size_t ...Dimensions>
void print(const Tensor<Dimensions...>& tensor, float numDecimals = 2.f);


template <std::size_t... Dimensions>
class Tensor 
{
    // Total size of the tensor
    static constexpr std::size_t _size = (1 * ... * Dimensions);

    // To make sure the tensor is not empty
    static_assert(_size > 0 && "Tensor size must be > 0");

    // The number of elements which cannot fit in a package
    static constexpr uint16_t _offset = _size % PACKAGE_LENGTH;

    static constexpr size_t UNROLL_FACTOR = (_size >= 16 * PACKAGE_LENGTH) ? 16 :
                                            (_size >= 8 * PACKAGE_LENGTH) ? 8 :
                                            (_size >= 4 * PACKAGE_LENGTH) ? 4 : 1;


public:


    // ------- TENSORS CONSTRUCTORS -------


    // Default constructor
    constexpr Tensor();

    // Copy constructor
    constexpr Tensor(const Tensor<Dimensions...>& other);

    // Rvalue assignement constructor
    constexpr Tensor(Tensor<Dimensions...>&& other) noexcept;

    // Assignement constructor for a different dimensions Tensor of the same size
    template <std::size_t... OtherDimensions>
    constexpr Tensor(const Tensor<OtherDimensions...>& other);

    // Rvalue assignement constructor for a different dimensions Tensor of the same size
    template <std::size_t... OtherDimensions>
    constexpr Tensor(Tensor<OtherDimensions...>&& other) noexcept;



    // Fill constructor with a given value
    constexpr Tensor(float value);

    // Fill the tensor from an initializer list
    constexpr Tensor(std::initializer_list<float> values);

    // Fill the tensor from a float array
    constexpr Tensor(float values[_size]);



    // Fill Tensors with zeros
    template <std::size_t... Dimensions>
    friend Tensor<Dimensions...> zeros();

    // Fill Tensors with ones
    template <std::size_t... Dimensions>
    friend Tensor<Dimensions...> ones();

    // Fill Tensors with random values with a distribution based on the given mean and the standard deviation
    template <std::size_t... Dimensions>
    friend Tensor<Dimensions...> normal(float mean, float std);



    // Fill Tensors with random values with the Glorot also named Xavier normal distribution
    // Often used with activation functions like tanh or sigmoid
    template <std::size_t numInput, std::size_t numOutput>
    friend  Tensor<numInput, numOutput> glorot_normal();

    // Fill Tensors with random values with the Glorot also named Xavier uniform distribution
    // Often used with activation functions like tanh or sigmoid
    template <std::size_t numInput, std::size_t numOutput>
    friend  Tensor<numInput, numOutput> glorot_uniform();


    // Fill Tensors with random values with He normal distribution
    // Often used with ReLU activation function
    template <std::size_t numInput, std::size_t numOutput>
    friend  Tensor<numInput, numOutput> he_normal();

    // Fill Tensors with random values with He uniform distribution
        // Often used with ReLU activation function
    template <std::size_t numInput, std::size_t numOutput>
    friend  Tensor<numInput, numOutput> he_uniform();



    // Fill Tensors with random values with Lecun normal distribution
    // Often used with SELU or sigmoid activation function
    template <std::size_t numInput, std::size_t numOutput>
    friend  Tensor<numInput, numOutput> lecun_normal();

    // Fill Tensors with random values with Lecun uniform distribution
    // Often used with SELU or sigmoid activation function
    template <std::size_t numInput, std::size_t numOutput>
    friend  Tensor<numInput, numOutput> lecun_uniform();


    // Destructor
    ~Tensor();


    // ------ TENSORS OPERATORS -------


    // Copy operator with a Tensor of the same dimensions
    constexpr void operator= (const Tensor<Dimensions...>& other);

    // Copy operator with a rvalue Tensor of the same dimensions
    constexpr void operator= (Tensor<Dimensions...>&& other) noexcept;



    // Element-wise addition with a float
    constexpr Tensor<Dimensions...> operator+(float value) const &;

    // Element-wise subtraction with a float
    constexpr Tensor<Dimensions...> operator-(float value) const &;

    // Element-wise multiplication with a float
    constexpr Tensor<Dimensions...> operator*(float value) const &;

    // Element-wise division with a float
    constexpr Tensor<Dimensions...> operator/(float value) const &;



    // Element-wise addition with a float
    constexpr Tensor<Dimensions...> operator+(float value) &&;

    // Element-wise subtraction with a float
    constexpr Tensor<Dimensions...> operator-(float value) &&;

    // Element-wise multiplication with a float
    constexpr Tensor<Dimensions...> operator*(float value) &&;

    // Element-wise division with a float
    constexpr Tensor<Dimensions...> operator/(float value) &&;



    // Element-wise addition with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator+(const Tensor<Dimensions...>& tensor) const &;

    // Element-wise subtraction with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator-(const Tensor<Dimensions...>& tensor) const &;

    // Element-wise multiplication with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator*(const Tensor<Dimensions...>& tensor) const &;

    // Element-wise division with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator/(const Tensor<Dimensions...>& tensor) const &;



    // Element-wise addition with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator+(const Tensor<Dimensions...>& tensor) &&;

    // Element-wise subtraction with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator-(const Tensor<Dimensions...>& tensor) &&;

    // Element-wise multiplication with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator*(const Tensor<Dimensions...>& tensor) &&;

    // Element-wise division with a Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator/(const Tensor<Dimensions...>& tensor) &&;



    // Element-wise addition with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator+(Tensor<Dimensions...>&& tensor) const &;

    // Element-wise subtraction with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator-(Tensor<Dimensions...>&& tensor) const &;

    // Element-wise multiplication with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator*(Tensor<Dimensions...>&& tensor) const &;

    // Element-wise division with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator/(Tensor<Dimensions...>&& tensor) const &;



    // Element-wise addition with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator+(Tensor<Dimensions...>&& tensor) &&;

    // Element-wise subtraction with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator-(Tensor<Dimensions...>&& tensor) &&;

    // Element-wise multiplication with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator*(Tensor<Dimensions...>&& tensor) &&;

    // Element-wise division with a rvalue Tensor of the same dimensions
    constexpr Tensor<Dimensions...> operator/(Tensor<Dimensions...>&& tensor) &&;



    // Element-wise addition of the tensor with another tensor of the same dimensions
    constexpr void operator+=(const Tensor<Dimensions...>& tensor);

    // Element-wise subtraction of the tensor with another tensor of the same dimensions
    constexpr void operator-=(const Tensor<Dimensions...>& tensor);

    // Element-wise multiplication of the tensor with another tensor of the same dimensions
    constexpr void operator*=(const Tensor<Dimensions...>& tensor);

    // Element-wise division of the tensor with another tensor of the same dimensions
    constexpr void operator/=(const Tensor<Dimensions...>& tensor);


    // ------ TENSORS ACCESSORS -------


    // Access the tensor values
    constexpr float& operator()(std::size_t index) const;

    constexpr bool operator==(const Tensor<Dimensions...>& tensor) const;

    
    // ------ TENSORS BATCHES OPERATORS -------


    // Element-wise addition with each tensors batches
    template<std::size_t batch_size>
    constexpr Tensor<Dimensions..., batch_size> operator+(const Tensor<Dimensions..., batch_size>& tensor) const;

    // Element-wise subtraction with each tensors batches
    template<std::size_t batch_size>
    constexpr Tensor<Dimensions..., batch_size> operator-(const Tensor<Dimensions..., batch_size>& tensor) const;

    // Element-wise multiplication with each tensors batches
    template<std::size_t batch_size>
    constexpr Tensor<Dimensions..., batch_size> operator*(const Tensor<Dimensions..., batch_size>& tensor) const;

    // Element-wise division with each tensors batches
    template<std::size_t batch_size>
    constexpr Tensor<Dimensions..., batch_size> operator/(const Tensor<Dimensions..., batch_size>& tensor) const;



    // Element-wise addition with each tensors batches rvalue Tensor
    template<std::size_t batch_size>
    constexpr Tensor<Dimensions..., batch_size> operator+(Tensor<Dimensions..., batch_size>&& tensor) const;

    // Element-wise subtraction with each tensors batches rvalue Tensor
    template<std::size_t batch_size>
    constexpr Tensor<Dimensions..., batch_size> operator-(Tensor<Dimensions..., batch_size>&& tensor) const;

    // Element-wise multiplication with each tensors batches rvalue Tensor
    template<std::size_t batch_size>
    constexpr Tensor<Dimensions..., batch_size> operator*(Tensor<Dimensions..., batch_size>&& tensor) const;

    // Element-wise division with each tensors batches rvalue Tensor
    template<std::size_t batch_size>
    constexpr Tensor<Dimensions..., batch_size> operator/(Tensor<Dimensions..., batch_size>&& tensor) const;


    // Element-wise addition of the tensor with each tensors batches
    template<std::size_t batch_size>
    constexpr void operator+=(const Tensor<Dimensions..., batch_size>& tensor);

    // Element-wise subtraction of the tensor with each tensors batches
    template<std::size_t batch_size>
    constexpr void operator-=(const Tensor<Dimensions..., batch_size>& tensor);

    // Sum of Element-wise multiplication of the tensor with each tensors batches
    template<std::size_t batch_size>
    constexpr void operator*=(const Tensor<Dimensions..., batch_size>& tensor);

    // Sum of Element-wise division of the tensor with each tensors batches
    template<std::size_t batch_size>
    constexpr void operator/=(const Tensor<Dimensions..., batch_size>& tensor);


    // ------ TENSORS BASIC FUNCTIONS -------


    // To print the tensor elements
    template<std::size_t ...Dimensions>
    friend void print(const Tensor<Dimensions...>& tensor, float numDecimals);

    // To get the shape of the tensor
    constexpr std::string shape() const;


    // Reshape the dimensions of the tensor to a compatible one
    template<std::size_t... newDimensions>
    constexpr Tensor<newDimensions...> reshape() const &;


    // Reshape the dimensions of the tensor to a compatible one for rvalue tensor
    template<std::size_t... newDimensions>
    constexpr Tensor<newDimensions...> reshape() &&;


    // Flatten the tensor to a one dimension tensor
    constexpr Tensor<(1 * ... * Dimensions)> flatten() const &;


    // Flatten the tensor to a one dimension tensor for rvalue tensor
    constexpr Tensor<(1 * ... * Dimensions)> flatten() &&;


    // Compute the absolute value of each element in the tensor
    template<std::size_t ...Dimensions>
    friend Tensor<Dimensions...> abs(const Tensor<Dimensions...>& tensor);

    
    // Compute the sum of each values in the tensor
    constexpr float sum() const;


    // Find the index of the maximum value in the tensor
    constexpr size_t argmax() const;


    // Find the maximum value in the tensor
    constexpr float max() const;


    // Compute the mean of the values the tensor
    constexpr float mean() const;


    // Compute the variance of the values in the tensor based on a given mean
    constexpr float variance(float mean) const;


    // ------ TENSORS MATRIX OPERATIONS -------


    // Element-wise addition between tensorC and the element-wise multiplication of tensorA and tensorB
    // @return tensorA * tensorB + tensorC
    template<std::size_t ...Dimensions>
    friend Tensor<Dimensions...> multiply_and_add(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC);


    // Element-wise subtraction between tensorC and the element-wise multiplication of tensorA and tensorB
    // @return tensorA * tensorB - tensorC
    template<std::size_t ...Dimensions>
    friend Tensor<Dimensions...> multiply_and_sub(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC);


    // Transpose the tensor
    template <std::size_t cols, std::size_t rows, std::size_t... rest>
    friend Tensor<rows, cols, rest...> transpose(const Tensor<cols, rows, rest...>& tensor);


    // Matrix multiplication between tensorA and tensorB
    template<std::size_t colsA, std::size_t rowsA, std::size_t colsB, std::size_t... rest>
    friend Tensor<colsB, rowsA, rest...> mul(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsB, colsA, rest...>& tensorB);


    // <!> WARNING : All the tensors dimensions are not taken into account <!> 
    // Matrix multiplication between tensorA and the transpose of tensorB
    template<std::size_t colsA, std::size_t rowsA, std::size_t colsB, std::size_t... rest>
    friend Tensor<colsB, rowsA, rest...> mul_transposed(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsA, colsB, rest...>& tensorB);


    // Matrix multiplication between tensorA and the transpose of tensorB as a scalar
    template<std::size_t colsA, std::size_t rowsA>
    friend Tensor<rowsA> mul_b_transposed_scalar(const Tensor<colsA, rowsA>& tensorA, const Tensor<colsA>& tensorB);


    // Matrix multiplication between the transpose of tensorA and tensorB both as a scalar
    template<std::size_t colsA, std::size_t colsB>
    friend Tensor<colsB, colsA> mul_transposed_scalar(const Tensor<colsA>& tensorA, const Tensor < colsB> & tensorB);


    // ------ TENSORS LAYERS FUNCTIONS -------


    // Apply the sigmoid function on the tensor values
    void apply_sigmoid();


    // Apply the ReLU function on the tensor values
    void apply_ReLU();

    // Apply the derivative of the ReLU function on the tensor values
    void apply_ReLU_derivative();


    // Apply a normal distribution on the tensor values
    void apply_normalization();

    // Normalize the tensor based on a given mean and variance then shift and scale the values
    void norm_shift_and_scale(float mean, float variance, float shift, float scale);


    // Apply dropout on the tensor values setting random values to zero based on a given rate
    // @return the build mask to be able to apply the same dropout on the backward pass
    Tensor<Dimensions...> apply_dropout(float rate);

    // Apply the given mask on the tensor values
    void apply_mask(const Tensor<Dimensions...>& mask);


    // Apply the softmax function on the tensor values
    void apply_Softmax();

    // Apply the derivative of the softmax function on the tensor values
    void apply_Softmax_derivative();

    // Compute the cross entropy loss function
    template<std::size_t ...Dimensions>
    friend float Cross_Entropy_loss(const Tensor<Dimensions...>& prediction, const Tensor<Dimensions...>& labels);


    /*template<PACKAGE_TYPE(*OP)(PACKAGE_TYPE, PACKAGE_TYPE), std::size_t ...Dimensions, std::size_t ...otherDimensions>
    friend void for_each(const Tensor<Dimensions..., otherDimensions...>& a, const Tensor<Dimensions...>& b, Tensor<Dimensions..., otherDimensions...>& c);*/


private:
    
    // Initialize the tensor values
    void init();

    // Split the tensor into multiple tensors of the given dimensions
    template<std::size_t... splittedDimensions>
    constexpr Tensor<splittedDimensions...>* split();


    // Pointer to the tensor values
    float* _values;

    // To get access to private members of Tensor with different dimensions
    template <std::size_t... otherDimensions>
    friend class Tensor;
};

#include "Tensor.cpp"
#include "network_utils.cpp"
#include "kernel_utils.cpp"
