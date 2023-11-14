#pragma once
#include "Layer.h"
#include "../network_parameters.h"

template <::std::size_t numInputNeurons, ::std::size_t numOutputNeurons>
class Dense : public Layer< Tensor<numInputNeurons>, Tensor<numOutputNeurons>>
{
    using inputType = Tensor<numInputNeurons>;
    using outputType = Tensor<numOutputNeurons>;

public:

    constexpr Dense() {}

    outputType Forward(inputType& input) override
    {      
        _previousInput = input;

        outputType output = mul_transposed_scalar(_weights, input) + _biases;

        return output;
    }

    inputType Backward(outputType& input) override
    {
        _weightsGradient += mul_transposed_scalar(input, _previousInput);
        _biasesGradient += input;

        inputType output = mul_transposed_scalar(transpose(_weights), input);

        return output;
    }

    void Update() override
    {
		_weights -= (_weightsGradient * learningRate + _previousWeightsGradient * momentum) / (float) batchSize;
        _biases -= (_biasesGradient * learningRate + _previousBiasesGradient * momentum) / (float) batchSize;

        _previousWeightsGradient = std::move(_weightsGradient);
        _previousBiasesGradient = std::move(_biasesGradient);

		_weightsGradient = zeros<numInputNeurons, numOutputNeurons>();
        _biasesGradient = zeros<numOutputNeurons>();
    }

private:

    Tensor<numInputNeurons, numOutputNeurons> _weights = normal<numInputNeurons, numOutputNeurons>(0.f, sqrt(1.f / numInputNeurons));
    Tensor<numOutputNeurons> _biases = zeros<numOutputNeurons>();

    inputType _previousInput;
    Tensor<numInputNeurons, numOutputNeurons> _weightsGradient = zeros<numInputNeurons, numOutputNeurons>();
    Tensor<numOutputNeurons> _biasesGradient = zeros<numOutputNeurons>();

    Tensor<numInputNeurons, numOutputNeurons> _previousWeightsGradient = zeros<numInputNeurons, numOutputNeurons>();
    Tensor<numOutputNeurons> _previousBiasesGradient = zeros<numOutputNeurons>();
};
