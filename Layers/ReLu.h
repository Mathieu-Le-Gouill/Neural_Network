#pragma once
#include "Layer.h"

template <::std::size_t... inputDims>
class ReLu : public Layer< Tensor<inputDims...>, Tensor<inputDims...>>
{
    using inputType = Tensor<inputDims...>;
    using outputType = Tensor<inputDims...>;

public:

    constexpr ReLu() {}

    outputType Forward(inputType& input) override
    {
        input.apply_ReLu();

        return std::move(input);
    }

    inputType Backward(outputType& input) override
    {
        input.apply_ReLu_derivative();

        return std::move(input);
    }

    void Update() override
    {
	}

};