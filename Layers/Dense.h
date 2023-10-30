#pragma once
#include "Layers/Layer.h"

template <size_t numInputNeurons, size_t numOutputNeurons>
class Dense : public Layer< Tensor<numInputNeurons>, Tensor<numOutputNeurons>>
{
    using inputType = Tensor<numInputNeurons>;
    using outputType = Tensor<numOutputNeurons>;

public:

    constexpr Dense() {}

    outputType Forward(inputType& input) override
    {
        outputType output = mul_transposed_scalar(_weights, input) + _biases;

        return output;
    }

    inputType Backward(outputType& input) override
    {
        inputType output;

        return output;
    }

private:

    Tensor<numInputNeurons, numOutputNeurons> _weights = rand<numInputNeurons, numOutputNeurons>(0.f, 1.f);
    Tensor<numOutputNeurons> _biases = zeros<numOutputNeurons>();
};
