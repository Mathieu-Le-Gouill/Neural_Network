#pragma once
#include "Layer.h"

template <::std::size_t... inputDims>
class Softmax : public Layer< Softmax<inputDims...>, Tensor<inputDims..., minibatchSize>, Tensor<inputDims..., minibatchSize>>
{
    using InputType = Tensor<inputDims..., minibatchSize>;
    using OutputType = Tensor<inputDims..., minibatchSize>;

public:

    constexpr Softmax() = default;

    inline OutputType Forward(InputType& input) noexcept
    {
        return input.apply_Softmax();
    }

    inline InputType Backward(OutputType& gradient) noexcept
    {
        return gradient.apply_Softmax_derivative();
    }


    inline void Update() noexcept
    {
    }

private:

};
