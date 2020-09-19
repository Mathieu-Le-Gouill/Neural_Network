#include "Neural_Network.h"
#include <math.h> 
#include <random>
#include <chrono> 
#include <iostream>

using namespace std;


Neural_Network::Neural_Network(const std::string& NeuralNetworkfileName, const double learningRate) : m_learningRate(learningRate)
{
	std::ifstream networkDataLoading;// File reader
	std::vector<unsigned short> networkTopology;// Vector will which contains the network topology
	std::vector<double> networkWeights, networkBiases;// Vector which will contains the weights and bias of the network

	GetData(networkTopology, networkWeights, networkBiases, networkDataLoading, NeuralNetworkfileName);// Obtain the network data from the file
	
	assert(networkTopology.size() >= 2 && "Error : the file network topology require a size greater than 2, for an input layer and an output layer.");// Be sure that the network topology have more than two layers

	unsigned nCounter = 0, wCounter = 0;// To count the neurons and weights during the loop

	for (unsigned l = 1; l < networkTopology.size(); l++)// For each neurons of the network, layer by layer except the first
		nCounter += networkTopology[l]; // Add the number of neurons in the layer
	for (unsigned l = 0; l < networkTopology.size() - 1; l++)// For each layer exept the last one
		wCounter += networkTopology[l] * networkTopology[l + 1]; // Add the links number of the layer

	assert(nCounter == networkBiases.size());// Be sure that the neuron number still the same than the biases number
	assert(wCounter == networkWeights.size());// Be sure that the neuron's links number still the same than the weights number
	
	nCounter = 0, wCounter = 0;// Reset their value for the next loop

	this->m_networktopology = networkTopology;


	for (unsigned l = 0; l < this->m_networktopology.size(); l++)// For each layer
	{
		this->m_outputs.push_back(Matrix<double>(this->m_networktopology[l]));// Add a matrix of neurons outputs
		this->m_weights.push_back(Matrix<double>());// Add a matrix of weights
		this->m_biases.push_back(Matrix<double>());// Add a matrix of biases

		std::vector<double> biases;// Create a new array of bias in the layer

		for (unsigned n = 0; n < this->m_networktopology[l]; n++)// For each neurons in the layer
		{
			if (l != 0)// If this is not the first layer
			{
				std::vector<double> weights;// Create a new array of weights in the layer

				biases.push_back(networkBiases[nCounter]);// Add the bias value to the array
				
				for (unsigned i = 0; i < this->m_networktopology[l - 1]; i++)// For each neurons in the previous layer
				{
					weights.push_back(networkWeights[wCounter]);// Set the neuron's weight value

					wCounter++;// Add one the the weights count
				}
				this->m_weights[l].add_a_Row(weights);// Set the bias value

				nCounter++;// Add one the the biases count
			}
		}
		if(!biases.empty())this->m_biases[l].add_a_Row(biases);// Set the neuron's bias value
	}
	this->m_errors = this->m_outputs;// Copy the matrix of outputs to the one of the errors
	
}


Neural_Network::Neural_Network(std::vector<unsigned short> networkTopology, const double learningRate) : m_learningRate(learningRate)// Constructor
{
	assert(networkTopology.size() >= 2 && "Error : the given network topology require a size greater than 2, for an input layer and an output layer.");// Be sure that the network topology have more than two layers

	m_networktopology = networkTopology;
	const unsigned &nbLayers = networkTopology.size();

	for (unsigned l = 0; l < nbLayers; l++)// For each layers of the network
	{
		this->m_outputs.push_back(Matrix<double>(networkTopology[l]));// Add a matrix of neurons outputs
		this->m_weights.push_back(Matrix<double>());// Add an empty matrix of neurons weights

		//- Set the weights values to randoms values
		const unsigned nbInputs = (l == 0) ? 0 : networkTopology[l - 1];// Get the number of neurons in the previous layer

		for (int n = 0; n < (int)networkTopology[l]; n++)// For each neurons of the layer
		{
			vector<double> neuron_weights;// Create a n empty array of weights for the neuron

			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();// To get differents epochs 
			std::default_random_engine generator(seed);// Create a generator of random numbers
			std::normal_distribution<double> distribution(0, 1);// Create a method of distribution of mean 0 and variance 1

			for (int i = 0; i < (int)nbInputs; i++)// For each neuron's inputs
			{
				double randomValue = distribution(generator);// Generate a random number by the generator using the distribution
				neuron_weights.push_back((double)(randomValue*sqrt(2.0 / nbInputs)));// Set neuron's weights using RELU
			}

			if(!neuron_weights.empty())	 m_weights.back().add_a_Row(neuron_weights);// Add weights to the neuron for each neurons inputs
		}
	}
	this->m_biases = this->m_errors = this->m_outputs;// Set the biases and the erros matrices
	this->m_biases.front() = Matrix<double>();// Remove the first layer biases because they won't be used

}


Neural_Network::~Neural_Network()// Destructor
{
}


void Neural_Network::FeedForward(const Matrix<double> &inputsValues)// Method to update the totality of the network according to the inputsValues
{
	assert(inputsValues.size() == m_networktopology[0] && inputsValues.rows() == 1 && "Error : the inputsValues array in the feedforward method must have the same size than the network toplogy's inputs layer");// Be sure that the number of inputs values size still the same than the network topology first layer

	//- Set the inputs values to the first layer of the network
	this->m_outputs[0] = inputsValues;// Set the inputs values into the first layer output

	//- Feed the other layers according to the inputs values
	for (unsigned l = 1; l < this->m_networktopology.size(); l++)// For each layers except the first
	{
		Matrix<double> Z = this->m_weights[l] * this->m_outputs[l - 1].transpose() + this->m_biases[l].transpose();
		this->m_outputs[l] = Sigmoid(Z).transpose();// Feed the output layer using weights and biases
	}

	//-To print the handwritten digits inputs from the MNIST file
		//for (int r=0; r < 28; r++){for (int c = 0; c < 28; c++)	cout << round(this->m_outputs[0][0][r * 28 + c])<<" ";	  cout << "" << endl;}
}


void Neural_Network::BackPropagate(const Matrix<double> &targetsValues)// Method to propagate the error obtained in the ouput layer to the hiddens
{
	assert(targetsValues.size() == this->m_networktopology.back());// Be sure that the number of targets values still the same than the network topology last layer

	//- Compute the output error between the outputs values in the last layer and the targets values

	Matrix<double> value(this->m_networktopology.back(), 1, 1);// Create a matrix of the neuron number in the last layer * 1, full of value equal to 1

	//- Compute the error in the last layer
	this->m_errors.back() = (this->m_outputs.back() - targetsValues).scalar(this->m_outputs.back().scalar(value - this->m_outputs.back()));


	//- Propagate the outputs errors from the last layer to the hiddens layers 
	for (int l = this->m_networktopology.size() - 2; l > 0; l--)// For each hiddens layers (excluding the output layer and the input layer)
	{
		value = Matrix<double>(this->m_networktopology[l], 1, 1);// Create a matrix of the neuron number in the layer * 1, full of value equal to 1

		this->m_errors[l] = (this->m_errors[l + 1] * this->m_weights[l + 1]).scalar(this->m_outputs[l].scalar(value - this->m_outputs[l]));// Backpropagate the error in the network using ouput error, weights and outputs
	}
}


void Neural_Network::Update()// Method to update the network according to the error got previously by the back propagation
{
	//- Update the weights and biases by computing gradients for each them
	for (unsigned l = 1; l < this->m_networktopology.size(); l++)// For each layers except the first
	{
		const Matrix<double> weightsGradient = this->m_errors[l].transpose() * this->m_outputs[l - 1];// Compute the weights gradients : same that (this->m_outputs[l - 1].transpose() * this->m_errors[l]).transpose()
		const Matrix<double> biasesGradient = this->m_errors[l];// Compute the biases gradients

		this->m_weights[l] -= weightsGradient * m_learningRate;// Update weights values
		this->m_biases[l] -= biasesGradient * m_learningRate; // Update biases values
	}
}


void Neural_Network::ShowResults(const int accuracy) const// Method to show the current output layer results
{
	this->m_outputs.back().print(accuracy);// Print the matrix values to the round to the nearest hundredth of the current neuron's output
}


void Neural_Network::SaveData(const std::string &fileName)// Method to save the current data progress of the network in a file
{
	std::ofstream networkDataSaving(fileName.c_str());// File writer

	if (networkDataSaving)// If it is open
	{
		// Save the network data in the file
		networkDataSaving << "topology:";// Network topology
		for (unsigned l = 0; l < this->m_networktopology.size(); l++)
			networkDataSaving << " " << m_networktopology[l];
		networkDataSaving << "\n" << std::endl;


		networkDataSaving << "weights:";// Network neurons weights
		for (unsigned l = 1; l < this->m_networktopology.size(); l++)
			if(!this->m_weights[l].empty())
			for (unsigned n = 0; n < this->m_networktopology[l]; n++)
				for (unsigned i = 0; i < this->m_networktopology[l - 1]; i++)
					networkDataSaving << " " << this->m_weights[l][n][i];
		networkDataSaving << "\n" << std::endl;


		networkDataSaving << "biases:";// Network neurons biases
		for (unsigned l = 0; l < this->m_networktopology.size(); l++)
			if (!this->m_biases[l].empty())
				for (unsigned n = 0; n < this->m_networktopology[l]; n++)
					networkDataSaving << " " << this->m_biases[l][0][n];
		networkDataSaving << "\n" << std::endl;
	}
	else // If it failed to open
		std::cout << "Error, can't oppening file network_data.txt" << std::endl;// Error

}


Matrix<double> Neural_Network::Sigmoid(Matrix<double> &matrix)// Sigmoid function for matrix
{
	for (unsigned r = 0; r < matrix.rows(); r++)
	{
		for (unsigned c = 0; c < matrix.columns(); c++)
		{
			matrix[r][c] = 1 / (1 + exp(-matrix[r][c]));// Compute the matrix sigmoid
		}
	}
	return matrix;
}


void Neural_Network::GetData(std::vector<unsigned short> &topology, std::vector<double> &weights, std::vector<double> &biases, std::ifstream &networkDataLoading, const std::string &NeuralNetworkfileName)// Method to recover a neural network in a file
{
	networkDataLoading = ifstream(NeuralNetworkfileName.c_str());// File reader

	if (networkDataLoading.is_open())// If we manage to open the file
	{

		std::string line; // Contains the lines
		std::string label; // Contais the words

		networkDataLoading.seekg(0, std::ios::beg);

		while (getline(networkDataLoading, line))// While there is a line didn't read yet in the file
		{
			std::stringstream ss(line);
			ss >> label;

			if (label.compare("topology:") == 0)// If the word topology appear
			{
				unsigned oneValue;
				while (ss >> oneValue)
				{
					topology.push_back(oneValue);// Add the neurons number to the topology array
				}
			}
			else if (label.compare("weights:") == 0)// If the word weights appear
			{
				double oneValue;
				while (ss >> oneValue)
				{
					weights.push_back(oneValue);// Add the neurons weights value to the weights array
				}
			}
			else if (label.compare("biases:") == 0)// If the word biases appear
			{
				double oneValue;
				while (ss >> oneValue)
				{
					biases.push_back(oneValue);// Add the neurons biases to the biases array
				}
			}
		}
	}
	else// If the file couldn't be opened
		cout << "Error during the loading of the network file" << NeuralNetworkfileName << endl;

}
