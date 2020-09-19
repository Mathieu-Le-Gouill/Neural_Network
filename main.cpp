#include "Matrix.h"
#include "Neural_Network.h"

using namespace std;

//VARIABLES
const unsigned short nbDataImage = 784;
const double learningRate = 1;
const unsigned nbEpochs = 1000;
const unsigned nbTrainingsImages = nbEpochs;// 60 000 max

//VARIABLES

//FUNCTIONS DECLARATIONS
vector<Matrix<double>> Load_MNIST_File(const string &MNIST_FilePath, int nbImages, int ImageDataSize);// Function to obtain the data inputs of the images from the MNIST training file
vector<Matrix<double>> GetTargetValues(const string &LabelFilePath, int nbImages);// Function 
//FUNCTIONS DECLARATIONS

int main()
{
	vector<unsigned short> networkTopology
	{nbDataImage, 20, 10};

	vector<Matrix<double>> inputsValues = Load_MNIST_File("train-images.idx3-ubyte", nbTrainingsImages, nbDataImage);// t10k-images.idx3-ubyte OR train-images.idx3-ubyte
	vector<Matrix<double>> targetsValues = GetTargetValues("train-labels.idx1-ubyte", nbTrainingsImages);// t10k-labels.idx1-ubyte OR train-labels.idx1-ubyte

	//Neural_Network net(networkTopology, learningRate);

	cout << "Loading of the netSave..." << endl;
	//Neural_Network net("netSave.txt");
	Neural_Network net(networkTopology);

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
	net.SaveData("netSave2.txt");

	system("pause");
}
