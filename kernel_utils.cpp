#pragma once
#include "Tensor.h"

constexpr enum Kernel_Initializer 
{
    Glorot_Normal,
    Glorot_Uniform,
    He_Normal,
    He_Uniform,
    LeCun_Normal,
    LeCun_Uniform,
    Zeros
};


namespace Tensor_Init
{

    // Fill Tensors with random values with the Glorot also named Xavier normal distribution
    template <::std::size_t numInput, ::std::size_t numOutput>
    inline Tensor<numInput, numOutput> glorot_normal()
    {
        return normal<numInput, numOutput>(0.0f, std::sqrtf(2.f / (float)(numInput + numOutput)));
    }



    // Fill Tensors with random values with the Glorot also named Xavier uniform distribution
    template <::std::size_t numInput, ::std::size_t numOutput>
    inline Tensor<numInput, numOutput> glorot_uniform()
    {
        return normal<numInput, numOutput>(0.0f, std::sqrtf(6.f / (float)(numInput + numOutput)));
    }


    // Fill Tensors with random values with He normal distribution
    template <::std::size_t numInput, ::std::size_t numOutput>
    inline Tensor<numInput, numOutput> he_normal()
    {
        return normal<numInput, numOutput>(0.0f, std::sqrtf(2.f / (float)(numInput)));
    }


    // Fill Tensors with random values with He uniform distribution
    template <::std::size_t numInput, ::std::size_t numOutput>
    inline Tensor<numInput, numOutput> he_uniform()
    {
        return normal<numInput, numOutput>(0.0f, std::sqrtf(6.f / (float)(numInput)));
    }


    // Fill Tensors with random values with Lecun normal distribution
    template <::std::size_t numInput, ::std::size_t numOutput>
    inline Tensor<numInput, numOutput> lecun_normal()
    {
        return normal<numInput, numOutput>(0.0f, std::sqrtf(1.f / (float)(numInput)));
    }


    // Fill Tensors with random values with Lecun uniform distribution
    template <::std::size_t numInput, ::std::size_t numOutput>
    inline Tensor<numInput, numOutput> lecun_uniform()
    {
        return normal<numInput, numOutput>(0.0f, std::sqrtf(3.f / (float)(numInput)));
    }
};


template<::std::size_t numInput, ::std::size_t numOutput, Kernel_Initializer kernel_initializer>
constexpr auto Kernel_init()
{
    if constexpr (kernel_initializer == Kernel_Initializer::Glorot_Normal) {
        return Tensor_Init::glorot_normal<numInput, numOutput>();
    }
    else if constexpr (kernel_initializer == Kernel_Initializer::Glorot_Uniform) {
        return Tensor_Init::glorot_uniform<numInput, numOutput>();
    }
    else if constexpr (kernel_initializer == Kernel_Initializer::He_Normal) {
        return Tensor_Init::he_normal<numInput, numOutput>();
    }
    else if constexpr (kernel_initializer == Kernel_Initializer::He_Uniform) {
        return Tensor_Init::he_uniform<numInput, numOutput>();
    }
    else if constexpr (kernel_initializer == Kernel_Initializer::LeCun_Normal) {
        return Tensor_Init::lecun_normal<numInput, numOutput>();
    }
    else if constexpr (kernel_initializer == Kernel_Initializer::LeCun_Uniform) {
        return Tensor_Init::lecun_uniform<numInput, numOutput>();
    }
    else {
        return zeros<numInput, numOutput>();
    }
}



