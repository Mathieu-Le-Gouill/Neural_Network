#include <iostream>
#include "Fully_Connected_Layer.h"
#include <chrono>
#include "MatrixF.h"

#define LOAD_NET 0 // Set to 1 to load a saved network
#define SAVE_NET 0 // Set to 1 to overwrite the network save
#define TEST_NET 1 // Set to 1 to test the network after training
#define TIME_NET 1 // Set to 1 to show the ellapsed time during training and testing
#define PERF_NET 1 // Set to 1 to show the performance of the network with loss and accuracy during training and testing 

using namespace std;

// VARIABLES
const uint16_t imageSize = 784;
const float learningRate = 0.01f;
const unsigned nbTestImages = 10000;// 10 000 max

const string netSavingPath = "netSave.txt";

const uint16_t fc_topology[] = { imageSize, 32, 10 };
const uint16_t fc_lenght = sizeof(fc_topology)/sizeof(uint16_t);

const unsigned nbEpochs = 20;
const uint16_t mini_batch_size = 32;
const unsigned nbTrainingsImages = nbEpochs * mini_batch_size;// 60 000 max

const ActivationFunction activation_function = SIGMOID;
// VARIABLES


// FUNCTIONS
MatrixF* Load_MNIST_File(const string& MNIST_FilePath, unsigned nbImages);
MatrixF* GetTargetValues(const string& LabelFilePath, unsigned nbImages);

int main()
{
	MatrixF* inputs = Load_MNIST_File("train-images.idx3-ubyte", nbTrainingsImages);
	MatrixF* targets = GetTargetValues("train-labels.idx1-ubyte", nbTrainingsImages);

	#if LOAD_NET
	Fully_Connected_Layer fc (netSavingPath, learningRate, activation_function);

	#else
	Fully_Connected_Layer fc((uint16_t*) fc_topology, fc_lenght, learningRate, activation_function);
	#endif

	float cumulativeAccuracy, cumulativeLoss;

	#if TIME_NET
	auto t_start = std::chrono::steady_clock::now();
	#endif

	cout << "TRAINING..." << endl;

	for (unsigned e = 0; e < nbEpochs; ++e)
	{
		#if PERF_NET
		cout << "Epoch " << e << " :" << endl;
		cumulativeAccuracy = cumulativeLoss = 0.f;
		#endif

		for (uint16_t b = 0; b < mini_batch_size; ++b)
		{
			const MatrixF& target = targets[e * mini_batch_size + b];
			MatrixF output, result;

			fc.Feed_Forward(inputs[e*mini_batch_size + b], result);

			const MatrixF& loss = result - target;

			fc.Back_Propagate(loss, output);

			#if PERF_NET
			cumulativeAccuracy += argmax(result) == argmax(target);
			cumulativeLoss += sum(abs(loss)) / fc_topology[fc_lenght - 1];
			#endif
		}

		#if PERF_NET
		cout << "Loss : " << cumulativeLoss / (float)mini_batch_size << endl;
		cout << "Accuracy : " << cumulativeAccuracy / (float)mini_batch_size << endl;
		#endif

		fc.Update(mini_batch_size);
	}

	#if TIME_NET
	auto t_end = std::chrono::steady_clock::now();
	cout << "\nTraining Time : " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << "ms" << endl;
	#endif

	delete[] inputs;
	delete[] targets;

	#if SAVE_NET
	cout << "\nSAVING..." << endl;
	fc.Save_Data(netSavingPath);
	#endif

	#if TEST_NET
	cout << "\nTESTING..." << endl;

	inputs = Load_MNIST_File("t10k-images.idx3-ubyte", nbTestImages);
	targets = GetTargetValues("t10k-labels.idx1-ubyte", nbTestImages);

	#if PERF_NET
	cumulativeAccuracy = cumulativeLoss = 0.f;
	#endif

	#if TIME_NET
	t_start = std::chrono::steady_clock::now();
	#endif

	for (unsigned i = 0; i < nbTestImages; ++i)
	{
		const MatrixF& target = targets[i];
		const MatrixF& input = inputs[i];

		MatrixF result;

		fc.Feed_Forward(input, result);

		const MatrixF& loss = result - target;

		#if PERF_NET
		cumulativeAccuracy += argmax(result) == argmax(target);
		cumulativeLoss += sum(abs(loss)) / fc_topology[fc_lenght - 1] ;
		#endif
	}
	#if PERF_NET
	cout << "Loss : " << cumulativeLoss / (float) nbTestImages << endl;
	cout << "Accuracy : " << cumulativeAccuracy / (float) nbTestImages << endl;
	#endif

	#if TIME_NET
	t_end = std::chrono::steady_clock::now();
	cout << "\nTesting Time : " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << "ms" << endl;
	#endif

	delete[] inputs;
	delete[] targets;
	#endif

	cout << "\COMPLETE !"<<endl;
	system("pause");
}
// FUNCTIONS