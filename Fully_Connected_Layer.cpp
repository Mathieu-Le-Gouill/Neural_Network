#include "Fully_Connected_Layer.h"
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include "Debug.h"

#define MOMENTUM 0.9f

/* TODO : 
	- add sheduler for increase accuracy results
	- add options for activation functions
	- use optimized algorithm to store and load data in binary
*/

Fully_Connected_Layer::Fully_Connected_Layer(uint16_t* const networkTopology, const uint16_t networkLength, const float learningRate, const ActivationFunction activationFunction)
	: m_networkTopology(networkTopology), m_networkLength(networkLength), m_learningRate(learningRate), m_activationFunction(activationFunction)
{
	uint16_t* iterTopology = m_networkTopology + 1;

	m_outputs = new MatrixF[networkLength];

	m_weights = new MatrixF[networkLength - 1];
	m_biases = new MatrixF[networkLength - 1];

	m_weightsGradients = new MatrixF[networkLength - 1];
	m_biasesGradients = new MatrixF[networkLength - 1];

	m_previousWeightsGradients = new MatrixF[networkLength - 1];
	m_previousBiasesGradients = new MatrixF[networkLength - 1];

	MatrixF* weights = m_weights;
	MatrixF* biases = m_biases;

	MatrixF* weightsGradients = m_weightsGradients;
	MatrixF* biasesGradients = m_biasesGradients;

	MatrixF* previousWeightsGradients = m_previousWeightsGradients;
	MatrixF* previousBiasesGradients = m_previousBiasesGradients;

	for (; iterTopology < networkTopology + networkLength; 
		++iterTopology, ++weights, ++biases, ++weightsGradients, ++biasesGradients, ++previousWeightsGradients, ++previousBiasesGradients)
	{
		const uint16_t& layerLength = *iterTopology;
		const uint16_t& previousLayerLength = *(iterTopology-1);

		if(m_activationFunction == RELU)
			*weights = rand(layerLength, previousLayerLength, 0.f, sqrt(2.f / (float) previousLayerLength));
		else if (m_activationFunction == SIGMOID || m_activationFunction == TANH)
			*weights = rand(layerLength, previousLayerLength, 0.f, sqrt(1.f / (float) previousLayerLength));
		else
			*weights = rand(layerLength, previousLayerLength, 0.f, 0.01);

		*biases = MatrixF(1, layerLength, 0.f);

		*weightsGradients = MatrixF(layerLength, previousLayerLength, 0.f);
		*biasesGradients = MatrixF(1, layerLength, 0.f);

		*previousWeightsGradients = MatrixF(layerLength, previousLayerLength, 0.f);
		*previousBiasesGradients = MatrixF(1, layerLength, 0.f);
	}
}


Fully_Connected_Layer::Fully_Connected_Layer(const std::string& networkfileName, const float learningRate, const ActivationFunction activationFunction) :
	m_learningRate(learningRate), m_networkTopology(nullptr), m_networkLength(0), m_activationFunction(activationFunction)
{
	std::ifstream networkLoader(networkfileName.c_str());// File reader

	float* weights = nullptr;
	float* biases = nullptr;

	uint16_t weightsLength = 0;
	uint16_t biasesLength = 0;

	if (networkLoader.is_open())
	{
		std::string line;
		std::string label;

		while (getline(networkLoader, line))
		{
			std::stringstream ss(line);
			ss >> label;

			if (label.compare("topology:") == 0)
			{
				ss >> m_networkLength;
				
				m_networkTopology = new uint16_t[m_networkLength];

				uint16_t* iter = m_networkTopology;
				while (ss >> *(iter++)) {}
			}

			else if (label.compare("weights:") == 0)// If the word weights appear
			{

				for (uint16_t* iterTopology = m_networkTopology; iterTopology < m_networkTopology + m_networkLength - 1; ++iterTopology)
				{
					const uint16_t& layerLength = *iterTopology;
					const uint16_t& nextLayerLength = *(iterTopology + 1);

					weightsLength += layerLength * nextLayerLength;
				}

				weights = new float[weightsLength];

				float* iterW = weights;
				while (ss >> *(iterW++)) {}
			}

			else if (label.compare("biases:") == 0)// If the word biases appear
			{
	

				for (uint16_t* iterTopology = m_networkTopology + 1; iterTopology < m_networkTopology + m_networkLength; ++iterTopology)
				{
					const uint16_t& layerLength = *iterTopology;

					biasesLength += layerLength;
				}

				biases = new float[biasesLength];

				float* iterB = biases;
				while (ss >> *(iterB++)) { }
			}

		}
	}

	m_outputs = new MatrixF[m_networkLength];

	m_weights = new MatrixF[m_networkLength - 1];
	m_biases = new MatrixF[m_networkLength - 1];

	m_weightsGradients = new MatrixF[m_networkLength - 1];
	m_biasesGradients = new MatrixF[m_networkLength - 1];

	m_previousWeightsGradients = new MatrixF[m_networkLength - 1];
	m_previousBiasesGradients = new MatrixF[m_networkLength - 1];

	uint16_t* iterTopology = m_networkTopology + 1;

	MatrixF* iterWeights = m_weights;
	MatrixF* iterBiases = m_biases;

	MatrixF* iterWeightsGradients = m_weightsGradients;
	MatrixF* iterBiasesGradients = m_biasesGradients;

	MatrixF* iterPreviousWeightsGradients = m_previousWeightsGradients;
	MatrixF* iterPreviousBiasesGradients = m_previousBiasesGradients;

	int weightsIndex = 0;
	int biasesIndex = 0;

	for (; iterTopology < m_networkTopology + m_networkLength;
		++iterTopology, ++iterWeights, ++iterBiases, ++iterWeightsGradients, ++iterBiasesGradients, ++iterPreviousWeightsGradients, ++iterPreviousBiasesGradients)
	{
		const uint16_t& layerLength = *iterTopology;
		const uint16_t& previousLayerLength = *(iterTopology - 1);

		float* iterWeightsValues = new float[layerLength * previousLayerLength];
		float* iterBiasesValues = new float[layerLength];

		memcpy(iterWeightsValues, weights + weightsIndex, layerLength * previousLayerLength * sizeof(float));
		memcpy(iterBiasesValues, biases + biasesIndex, layerLength * sizeof(float));

		*iterBiases = MatrixF(1, layerLength, iterBiasesValues);

		*iterWeights = MatrixF(layerLength, previousLayerLength, iterWeightsValues);

		*iterWeightsGradients = MatrixF(layerLength, previousLayerLength, 0.f);
		*iterBiasesGradients = MatrixF(1, layerLength, 0.f);

		*iterPreviousWeightsGradients = MatrixF(layerLength, previousLayerLength, 0.f);
		*iterPreviousBiasesGradients = MatrixF(1, layerLength, 0.f);

		weightsIndex += layerLength * previousLayerLength;
		biasesIndex += layerLength;
	}

}


Fully_Connected_Layer::~Fully_Connected_Layer()
{
	delete[] m_networkTopology, m_weights, m_biases, m_outputs;
	delete[] m_weightsGradients, m_biasesGradients;
	delete[] m_previousWeightsGradients, m_previousBiasesGradients;
}


void Fully_Connected_Layer::Feed_Forward(const MatrixF& input, MatrixF& output)
{
	debug_assert(input.cols() == m_networkTopology[0] && input.rows() == 1 && "Error in Fully_Connected_Layer Feed_Forward the given input does not have a correct size.");
	
	*m_outputs = input;

	MatrixF* weights = m_weights;
	MatrixF* biases= m_biases;
	MatrixF* outputs = m_outputs;

	for (; weights < m_weights + m_networkLength - 1;
		++weights, ++biases)
	{
		// Propagate the input through the network
		MatrixF z = dot(*weights, transpose(*outputs)) + transpose(*biases);

		*(++outputs) = transpose(sigmoid(z));
	}

	output = *outputs;
}


void Fully_Connected_Layer::Back_Propagate(const MatrixF& upstreamLossGradient, MatrixF& lossGradient)
{
	debug_assert(upstreamLossGradient.cols() == m_networkTopology[m_networkLength - 1] && upstreamLossGradient.rows() == 1 && "Error in Fully_Connected_Layer Back_Propagate the given input does not have a correct size.");

	MatrixF loss = upstreamLossGradient * m_outputs[m_networkLength - 1] * (MatrixF(1,upstreamLossGradient.cols(), 1.f) - m_outputs[m_networkLength - 1]); // Loss * activationFunctionDerivative

	MatrixF* weights = m_weights + m_networkLength - 2;
	MatrixF* biases = m_biases + m_networkLength - 2;

	MatrixF* weightsGradients = m_weightsGradients + m_networkLength - 2;
	MatrixF* biasesGradients = m_biasesGradients + m_networkLength - 2;

	MatrixF* outputs = m_outputs + m_networkLength - 2;

	for (; weights >= m_weights;
		--weights, --biases, --outputs, --weightsGradients, --biasesGradients)
	{
		// - Update the Gradients
		*weightsGradients += dot(transpose(loss), *outputs);
		*biasesGradients += loss;

		// - Backpropagate the error in the network
		loss = dot(loss, *weights) * *outputs * (MatrixF(1, outputs->cols(), 1.f) - *outputs); // L * W * activation_function_derivative(O)
	}

	lossGradient = loss;
}

void Fully_Connected_Layer::Update(const uint16_t miniBatchSize)
{
	MatrixF* weights = m_weights;
	MatrixF* biases = m_biases;

	MatrixF* weightsGradients = m_weightsGradients;
	MatrixF* biasesGradients = m_biasesGradients;

	MatrixF* previousWeightsGradients = m_previousWeightsGradients;
	MatrixF* previousBiasesGradients = m_previousBiasesGradients;
	
	for (; weights < m_weights + m_networkLength - 1; 
		++weights, ++biases, ++weightsGradients, ++biasesGradients, ++previousWeightsGradients, ++previousBiasesGradients)
	{
		// - Update the Weights and Biases
		*weights -= (*weightsGradients * m_learningRate + *previousWeightsGradients * MOMENTUM) / (float) miniBatchSize;
		*biases -= (*biasesGradients * m_learningRate + *previousBiasesGradients * MOMENTUM) / (float) miniBatchSize;

		// - Keep the gradients for next momentum
		*previousWeightsGradients = *weightsGradients;
		*previousBiasesGradients = *biasesGradients;

		// - Reset the Gradients
		fill(*weightsGradients, 0.f);
		fill(*biasesGradients, 0.f);
	}
}


void Fully_Connected_Layer::Show_Results(float decimals) const
{
	print(m_outputs[m_networkLength - 1], decimals);
}


void Fully_Connected_Layer::Save_Data(const std::string& fileName) const
{
	std::ofstream networkWritter(fileName.c_str());// File writer

	uint16_t* iterTopology = m_networkTopology;

	MatrixF* iterWeights = m_weights;
	MatrixF* iterBiases = m_biases;

	if (networkWritter)
	{
		networkWritter << "topology:";
		networkWritter << " " << m_networkLength;
		for (; iterTopology < m_networkTopology + m_networkLength; ++iterTopology)
			networkWritter << " " << *iterTopology;
		networkWritter << "\n";

		networkWritter << "weights:";
		for (; iterWeights < m_weights + m_networkLength - 1; ++iterWeights)
			for (uint16_t r = 0; r < iterWeights->rows(); ++r)
				for (uint16_t c = 0; c < iterWeights->cols(); ++c)
					networkWritter << " " << iterWeights->operator()(r, c);
		networkWritter << "\n";

		networkWritter << "biases:";
		for (; iterBiases < m_biases + m_networkLength - 1; ++iterBiases)
			for (uint16_t c = 0; c < iterBiases->cols(); ++c)
				networkWritter << " " << iterBiases->operator()(0, c);
		networkWritter << "\n";
	}
	else
		std::cout << "Error in Fully_Connected_Layer Save_Data failed to load the given data file" << std::endl;
}