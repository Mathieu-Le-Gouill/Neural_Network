#pragma once
#include <iostream>
#include "MatrixF.h"
#include <string.h>

enum class ActivationFunction { ReLU = 0, Sigmoid = 1, TanH = 2 };

#define RELU ActivationFunction::ReLU 
#define SIGMOID ActivationFunction::Sigmoid
#define TANH ActivationFunction::TanH


class Fully_Connected_Layer
{
public:

	Fully_Connected_Layer(uint16_t* const networkTopology, const uint16_t networkLength, const float learningRate, const ActivationFunction activationFunction);

	Fully_Connected_Layer(const std::string& networkfileName, const float learningRate, const ActivationFunction activationFunction);

	~Fully_Connected_Layer();

	void Feed_Forward(const MatrixF& input, MatrixF& output);

	void Back_Propagate(const MatrixF& upstreamLossGradient, MatrixF& lossGradient);

	void Update(const uint16_t miniBatchSize = 1);

	void Show_Results(float decimals = 2.f) const;

	void Save_Data(const std::string &fileName) const;

private:

	const float m_learningRate;

	ActivationFunction m_activationFunction;

	uint16_t* m_networkTopology;
	uint16_t m_networkLength;

	MatrixF* m_outputs;

	MatrixF* m_weights;
	MatrixF* m_biases;

	MatrixF* m_weightsGradients;
	MatrixF* m_biasesGradients;

	MatrixF* m_previousWeightsGradients;
	MatrixF* m_previousBiasesGradients;
};

