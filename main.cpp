#include <iostream>
#include <chrono>
#include "Pipeline.h"
#include "MNIST_Loader.h"

#define PERF_NET 0
#define TEST_NET 1
#define TIME_NET 1

using namespace ::std;
using namespace ::std::chrono;

int main()
{  
    auto inputs = Load_MNIST_File("train-images.idx3-ubyte", nbTrainImages);
    auto targets = GetTargetValues("train-labels.idx1-ubyte", nbTrainImages);

    Pipeline< Flatten<inputWidth, inputHeight>,
        Dense<inputSize, 32, LeCun_Normal>,
        Sigmoid<32>,
        Dense<32, 10, LeCun_Normal>,
        Sigmoid<10> > network;

    float cumulativeAccuracy;
    float cumulativeLoss;

    std::cout << "TRAINING...\n";

    #if TIME_NET
        auto t_start = std::chrono::steady_clock::now();
    #endif

    for (size_t e = 0; e < epochs; ++e) 
    {
        #if PERF_NET
            std::cout << "\nEpoch " << e << " :" << endl;
            cumulativeAccuracy = 0.f;
            cumulativeLoss = 0.f;
        #endif

        for (size_t b = 0; b < minibatchSize; ++b) 
        {
            const auto& input = inputs[e * minibatchSize + b];
            const auto& target = targets[e * minibatchSize + b];
            
            Tensor<outputSize> prediction = network.forward(input);

            Tensor<outputSize> error = prediction - target;

            network.backward(error);

            #if PERF_NET
                cumulativeAccuracy += prediction.argmax() == target.argmax();
                cumulativeLoss += (abs(error)).sum() / static_cast<float>(outputSize);
            #endif
        }

        #if PERF_NET
            std::cout << "Loss : " << cumulativeLoss / static_cast<float>(minibatchSize) << endl;
            std::cout << "Accuracy : " << cumulativeAccuracy / static_cast<float>(minibatchSize) << "\n" << endl;
        #endif

        network.update();
    }

    #if TIME_NET
        auto t_end = std::chrono::steady_clock::now();
        cout << "\nTraining Time : " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << "ms" << endl;
    #endif

    delete[] inputs; 
    delete[] targets;

    #if TEST_NET
        cout << "\nTESTING..." << endl;

        inputs = Load_MNIST_File("t10k-images.idx3-ubyte", nbTestImages);
        targets = GetTargetValues("t10k-labels.idx1-ubyte", nbTestImages);

        cumulativeAccuracy = cumulativeLoss = 0.f;

        for (size_t i = 0; i < nbTestImages; ++i)
        {
            const auto& target = targets[i];
            const auto& input = inputs[i];

            Tensor<outputSize> prediction = network.forward(input);

            Tensor<outputSize> error = prediction - target;

            cumulativeAccuracy += prediction.argmax() == target.argmax();
            cumulativeLoss += (abs(error)).sum() / static_cast<float>(outputSize);
        }

        cout << "Loss : " << cumulativeLoss / (float)nbTestImages << endl;
        cout << "Accuracy : " << cumulativeAccuracy / (float)nbTestImages << endl;

        delete[] inputs;
        delete[] targets;

    #endif

    cout << "\nCOMPLETE !" << endl;
    return 0;
}