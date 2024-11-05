#pragma once
#include "Layer.h"

template <::size_t... inputDims>
class Dropout : public Layer< Tensor<inputDims..., minibatchSize>, Tensor<inputDims..., minibatchSize>>
{
    using inputType = Tensor<inputDims..., minibatchSize>;
    using outputType = Tensor<inputDims..., minibatchSize>;

public:

    constexpr Dropout(float rate) : _rate(rate) {}

    inline outputType Forward(inputType& input) override
    {
        _mask = input.applyDropout(rate);

        return input;
    }

    inline inputType Backward(outputType& input) override
    {
        input.apply_mask(_mask);

        return input;
    }

    inline void Update() override
    {
    }

private:

    float _rate;
    Tensor<inputDims...> _mask;
};
