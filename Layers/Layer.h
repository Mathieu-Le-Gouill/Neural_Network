#pragma once
#include "Tensor.h"

template <typename input, typename output>
class Layer
{
public:

    virtual output Forward(input& tensor) = 0;

    virtual input Backward(output& tensor) = 0;
};

