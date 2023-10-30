#include <iostream>
#include <chrono>
#include "Layers/Conv.h"
#include "Layers/ReLu.h"
#include "Pipeline.h"
#include "Layers/Dense.h"
#include "Layers/Flatten.h"

using namespace std;
using namespace std::chrono;
#include <immintrin.h>



int main() 
{
    // Define input tensor dimensions
    constexpr size_t inputWidth = 5;
    constexpr size_t inputHeight = 5;
    constexpr size_t inputChannels = 3;


    Pipeline< ReLu<3,3,3>,
              ReLu<1,1,3>,
              Flatten<1,1,3>,
              Dense<3, 2> > pipe;

    //auto output = pipe.forward(Tensor<inputWidth, inputHeight, inputChannels>(1.f));
    auto output = pipe.backward(Tensor<2>(1.f));
    print(output);


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