#pragma once
#include "Layer.h"

template <::std::size_t... inputDims>
class ReLu : public Layer< Tensor<inputDims...>, Tensor<inputDims...>>
{
    using inputType = Tensor<inputDims...>;
    using outputType = Tensor<inputDims...>;

public:

    constexpr ReLu() {}

    inline outputType Forward(inputType& input) override
    {
        input.apply_ReLu();

        return input;
    }

    inline inputType Backward(outputType& input) override
    {
        input.apply_ReLu_derivative();

        return std::move(input);
    }

    inline void Update() override
    {
	}

};
