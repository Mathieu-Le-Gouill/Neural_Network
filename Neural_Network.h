#pragma once
#include "assert.h"
#include <fstream>
#include <sstream>
#include <string>

#include <vector>
#include "Matrix.h"


class Neural_Network
{
public:

	Neural_Network(const std::string& NeuralNetworkfileName, const double learningRate = 1);// Constructor using a network from file
	Neural_Network(std::vector<unsigned short> networkTopology, const double learningRate = 1);// Constructor to create a new network
	~Neural_Network();// Destructor

	void FeedForward(const Matrix<double> &inputsValues);// Method to update the totality of the network according to the inputsValues
	void BackPropagate(const Matrix<double> &targetsValues);// Method to propagate the error obtained in the ouput layer to the hiddens
	void Update();// Method to update the network according to the error got previously by the back propagation

	void ShowResults(const int accuracy=100) const;// Method to show the current output layer results
	void SaveData(const std::string &fileName);// Method to save the current data progress of the network in a file

private:

	static Matrix<double> Sigmoid(Matrix<double> &matrix);// Sigmoid function for matrix

	void GetData(std::vector<unsigned short> &topology, std::vector<double> &weights, std::vector<double> &biases, std::ifstream &networkDataLoading, const std::string &NeuralNetworkfileName);// Method to recover a neural network in a file


	std::vector<unsigned short> m_networktopology;// Contains the topology of the network
	std::vector<Matrix<double>> m_outputs;// Contains the neurons activation
	std::vector<Matrix<double>> m_biases;// Contains the neurons biases
	std::vector<Matrix<double>> m_weights;// Contains the neurons weights
	std::vector<Matrix<double>> m_errors;// Contains the errors values

	const double m_learningRate;// Coefficiant corresponding to the learning speed at each epoch

};

