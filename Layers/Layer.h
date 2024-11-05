#pragma once
#include "Tensor.h"
#include "../network_parameters.h"

template <typename child, typename input, typename output>
class Layer
{
public:

    constexpr virtual ~Layer() = default;

    inline auto Forward(input& tensor)
    {
        return static_cast<child*>(this)->Forward(tensor);
    }

    inline auto Backward(output& tensor)
    {
        return static_cast<child*>(this)->Backward(tensor);
    }

    inline void Update() 
    {
        static_cast<child*>(this)->Update();
    }
};

