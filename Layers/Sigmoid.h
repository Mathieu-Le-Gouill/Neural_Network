#pragma once
#include "Layer.h"

template <::std::size_t... inputDims>
class Sigmoid : public Layer< Tensor<inputDims...>, Tensor<inputDims...>>
{
    using inputType = Tensor<inputDims...>;
    using outputType = Tensor<inputDims...>;

public:

    constexpr Sigmoid() {}

    inline outputType Forward(inputType& input) override
    {
        input.apply_sigmoid();

        _output = input;

        return input;
    }

    inline inputType Backward(outputType& input) override
    {
        input *= _output * (ones<inputDims...>() - _output);

        return std::move(input);
    }

    void Update() override
    {
    }

private :
    outputType _output;

};
