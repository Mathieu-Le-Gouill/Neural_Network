#pragma once
#include "Layer.h"

template <::std::size_t... inputDims>
class Flatten : public Layer< Flatten<inputDims...>, Tensor<inputDims...>, Tensor<(1 * ... * inputDims)>>
{
    static constexpr ::std::size_t outputSize = (1 * ... * inputDims);

    using inputType = Tensor<inputDims...>;
    using outputType = Tensor<outputSize>;

public:

    constexpr Flatten() {}

    inline outputType Forward(inputType& input)
    {
        return std::move(input).flatten();
    }

    inline inputType Backward(outputType& input)
    {
        return std::move(input).template reshape<inputDims...>();
    }

    inline void Update()
    {
    }
};
