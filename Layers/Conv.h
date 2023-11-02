#pragma once
#include "Layer.h"


struct Kernel
{
    uint16_t width;
    uint16_t height;
};

struct Stride
{
    uint16_t horizontal;
    uint16_t vertical;
};

template <Kernel k, Stride s, size_t first, size_t second, size_t... inputDims>
class Conv : public Layer< Tensor<first, second, inputDims...>, Tensor<(first - k.width) / s.horizontal + 1, (second - k.height) / s.vertical + 1, inputDims...>>
{
    using inputType = Tensor<first, second, inputDims...>;
    using outputType = Tensor<(first - k.width) / s.horizontal + 1, (second - k.height) / s.vertical + 1, inputDims...>;

public:
    
    constexpr Conv() {}

    outputType Forward(inputType& tensor) override
    {
        outputType output(3.f);

        return output;
    }

    inputType Backward(outputType& tensor) override
    {
        inputType output(3.f);

        return output;
    }

    void Update() override
    {
	}

private:

    Tensor<k.width, k.height> _kernel = rand<k.width, k.height>(0.f, 1.f);   
};
