#include "Matrix.h"
#include "Neural_Network.h"
#include <iostream>


using namespace std;

//VARIABLES
const unsigned short nbDataImage = 784;
const double learningRate = 1;
const unsigned nbEpochs = 5000;
const unsigned nbTrainingsImages = nbEpochs;// 60 000 max

//VARIABLES

//FUNCTIONS DECLARATIONS
vector<Matrix> Load_MNIST_File(const string& MNIST_FilePath, int nbImages, int ImageDataSize);// Function to obtain the data inputs of the images from the MNIST training file
vector<Matrix> GetTargetValues(const string& LabelFilePath, int nbImages);// Function to obtain the desired output for each images
//FUNCTIONS DECLARATIONS

int main()
{
	vector<unsigned short> networkTopology
	{ nbDataImage, 20, 10 };

	vector<Matrix> inputsValues = Load_MNIST_File("t10k-images.idx3-ubyte", nbTrainingsImages, nbDataImage);// t10k-images.idx3-ubyte OR train-images.idx3-ubyte
	vector<Matrix> targetsValues = GetTargetValues("t10k-labels.idx1-ubyte", nbTrainingsImages);// t10k-labels.idx1-ubyte OR train-labels.idx1-ubyte

	Neural_Network net(networkTopology, learningRate);

	//cout << "Loading of the netSave..." << endl;
	//Neural_Network net("netSave.txt");

	//-Network training process 
	for (int e = 0; e < nbEpochs; e++)// For each epochs
	{
		cout << "Epoch " << e << " :" << endl;
		net.FeedForward(inputsValues[e]);
		net.BackPropagate(targetsValues[e]);
		cout << "Targets : "; targetsValues[e].print();
		net.Update();
		cout << "Results : "; net.ShowResults();
	}

	cout << "Saving..." << endl;
	net.SaveData("netSave.txt");

	system("pause");
}
