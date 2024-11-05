#pragma once
#include "Layer.h"

template <::std::size_t... inputDims>
class ReLU : public Layer< ReLU<inputDims...>, Tensor<inputDims...>, Tensor<inputDims...>>
{
    using inputType = Tensor<inputDims...>;
    using outputType = Tensor<inputDims...>;

public:

    constexpr ReLU() {}

    inline outputType Forward(inputType& input) noexcept
    {
        inputType output = std::move(input);
        output.apply_ReLU();

        return output;
    }

    inline inputType Backward(outputType& input)  noexcept
    {
        outputType output = std::move(input);
        output.apply_ReLU_derivative();

        return output;
    }

    inline void Update() noexcept
    {
	}

};
