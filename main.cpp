#include <iostream>
#include <chrono>
#include "Pipeline.h"

using namespace ::std;
using namespace ::std::chrono;
#include <immintrin.h>

int main() 
{
    Pipeline< Flatten<inputWidth, inputHeight, inputChannels>,
              Dense<inputSize, 20>,
              ReLu<20>,
              Dense<20, 10>,
              ReLu<10> > network;

    std::cout<<"Forward pass: \n";
    auto output = network.forward(rand<inputWidth, inputHeight, inputChannels>(0.f, 1.f));
    print(output);

    std::cout << "Backward pass: \n";
    auto output2 = network.backward(rand<outputSize>(0.f, 1.f));
    print(output2);


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