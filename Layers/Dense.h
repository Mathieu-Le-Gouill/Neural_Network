#pragma once
#include "Layer.h"

template <::std::size_t num_input_neurons,
          ::std::size_t num_output_neurons,
          Kernel_Initializer kernel_initializer = Kernel_Initializer::Glorot_Normal>

class Dense : public Layer< Dense<num_input_neurons, num_output_neurons, kernel_initializer>,
                            Tensor<num_input_neurons>,
                            Tensor<num_output_neurons>>
{
    using InputType = Tensor<num_input_neurons>;
    using OutputType = Tensor<num_output_neurons>;

public:

    constexpr Dense() = default;

    inline OutputType Forward(InputType& input) noexcept
    {   
        // Compute : Weights * Input + Biases
        OutputType output = mul_b_transposed_scalar(_weights, input) + _biases;

        _previousInput = std::move(input); // Cache the input for backpropagation

        return output;
    }

    inline InputType Backward(OutputType& upstream_loss_gradient) noexcept
    {
        // Calculate gradients with respect to weights and biases
        _weights_gradient += mul_transposed_scalar(upstream_loss_gradient, _previousInput);
        _biases_gradient  += upstream_loss_gradient;

        // Compute loss gradient with respect to the input (previous layer)
        InputType loss_gradient = mul(upstream_loss_gradient.template reshape<num_output_neurons, 1>(), _weights);


        return loss_gradient;
    }


    inline void Update() noexcept
    {
        float scaleFactor = 1.0f / static_cast<float>(minibatchSize);

        // Update the velocity
        _weights_velocity = (_weights_velocity * momentum - _weights_gradient * learningRate) * scaleFactor;
        _biases_velocity  = (_biases_velocity  * momentum - _biases_gradient * learningRate) * scaleFactor;

        // Update the weights and biases
        _weights += _weights_velocity;
        _biases  += _biases_velocity;

        // Reset the gradient
        _weights_gradient = zeros<num_input_neurons, num_output_neurons>();
        _biases_gradient = zeros<num_output_neurons>();
    }


private:

    Tensor<num_input_neurons, num_output_neurons> _weights = Kernel_init<num_input_neurons, num_output_neurons, kernel_initializer>();
    Tensor<num_output_neurons> _biases = zeros<num_output_neurons>();

    InputType _previousInput;

    Tensor<num_input_neurons, num_output_neurons> _weights_gradient = zeros<num_input_neurons, num_output_neurons>();
    Tensor<num_output_neurons> _biases_gradient = zeros<num_output_neurons>();

    Tensor<num_input_neurons, num_output_neurons> _weights_velocity = zeros<num_input_neurons, num_output_neurons>();
    Tensor<num_output_neurons> _biases_velocity = zeros<num_output_neurons>();
};
