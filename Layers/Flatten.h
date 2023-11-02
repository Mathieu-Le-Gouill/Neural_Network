#pragma once
#include "Layer.h"

template <::std::size_t... inputDims>
class Flatten : public Layer< Tensor<inputDims...>, Tensor<(1 * ... * inputDims)>>
{
    static constexpr ::std::size_t outputSize = (1 * ... * inputDims);

    using inputType = Tensor<inputDims...>;
    using outputType = Tensor<outputSize>;

public:

    constexpr Flatten() {}

    outputType Forward(inputType& input) override
    {
        return input.flatten();
    }

    inputType Backward(outputType& input) override
    {
        return input.reshape<inputDims...>();
    }

    void Update() override
    {
    }
};
