#include <iostream>
#include <chrono>
#include "Pipeline.h"

using namespace ::std;
using namespace ::std::chrono;
#include <immintrin.h>

int main() 
{
    /*auto a = rand<3, 5>(0.f, 1.f);
    std::cout<< "a: \n";
    print(a);

    std::cout << "argmax: \n";
    cout << a.argmax();*/


    Pipeline< Flatten<inputWidth, inputHeight, inputChannels>,
              Dense<inputSize, 20>,
              Sigmoid<20>,
              Dense<20, 10>,
              Sigmoid<10> > network;

    float cumulativeAccuracy = 0.f;
    float cumulativeLoss = 0.f;

    cout << "TRAINING...\n";

    for (size_t e = 0; e < epochs; ++e)
    {
        std::cout << "Epoch: " << e << "\n";

        Tensor<inputWidth, inputHeight, inputChannels> input = normal<inputWidth, inputHeight, inputChannels>(0.f, 1.f);

        //std::cout << "Forward pass: \n";
        Tensor<outputSize> result = network.forward(input);

        std::cout << "Result: \n";
        print(result);

        Tensor<outputSize> target = normal<outputSize>(0.f, 1.f);
        Tensor<outputSize> loss = result - target;

        std::cout << "Loss: \n";
        print(loss);

        //std::cout << "Backward pass: \n";
        auto upstreamLoss = network.backward(loss);

        network.update();

        cumulativeAccuracy += result.argmax() == target.argmax();

        cumulativeLoss += abs(loss).sum() / outputSize;
    }

    cout << "TESTING...\n";


    /*Tensor<9, 9> a = rand<9, 9>();

    Tensor<9, 9> b = rand<9, 9>();

    Tensor<9, 9> b_transposed = transpose(b);

    cout << "b :" << endl;
    print(b);

    cout << "\ntransposed of b :" << endl;
    print(b_transposed);


    cout << "\nmul :" << endl;
    print(mul(a, b));

    cout << "\nmul_transposed :" << endl;
    print(mul_transposed(a, b_transposed));*/


    system("pause");
    return 0;
}