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
        _previousInput = std::move(input);
        outputType output = mul_transposed_scalar(_weights, _previousInput) + _biases;

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
		_weights -= (_weightsGradient * learningRate + std::move(_previousWeightsGradient) * momentum) / (float) batchSize;
        _biases -= (_biasesGradient * learningRate + std::move(_previousBiasesGradient) * momentum) / (float) batchSize;

        _previousWeightsGradient = std::move(_weightsGradient);
        _previousBiasesGradient = std::move(_biasesGradient);

		_weightsGradient = zeros<numInputNeurons, numOutputNeurons>();
        _biasesGradient = zeros<numOutputNeurons>();
    }

private:

    Tensor<numInputNeurons, numOutputNeurons> _weights = rand<numInputNeurons, numOutputNeurons>(0.f, 1.f);
    Tensor<numOutputNeurons> _biases = zeros<numOutputNeurons>();

    inputType _previousInput;
    Tensor<numInputNeurons, numOutputNeurons> _weightsGradient = zeros<numInputNeurons, numOutputNeurons>();
    Tensor<numOutputNeurons> _biasesGradient = zeros<numOutputNeurons>();

    Tensor<numInputNeurons, numOutputNeurons> _previousWeightsGradient = zeros<numInputNeurons, numOutputNeurons>();
    Tensor<numOutputNeurons> _previousBiasesGradient = zeros<numOutputNeurons>();
};
