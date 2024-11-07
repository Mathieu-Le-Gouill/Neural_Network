#pragma once
#include "Layer.h"
#include "../network_parameters.h"

template <::std::size_t... inputDims>
class Norm : public Layer< Norm<inputDims...>, Tensor<inputDims..., minibatchSize>, Tensor<inputDims..., minibatchSize>>
{
	using inputType = Tensor<inputDims..., minibatchSize>;
	using outputType = Tensor<inputDims..., minibatchSize>;

	public:
		constexpr Norm() {}

		outputType Forward(inputType& input) override
		{
			_mean = input.mean();
			_variance = input.variance(_mean);

			input.norm_shift_and_scale(_mean, _variance, _shift, _scale);

			return std::move(input);
		}

		inputType Backward(outputType& input) override
		{
			_shiftGradient += input.sum();
			_scaleGradient += _shiftGradient * _scale;

			input *= _scale / sqrtf(_variance);

			return std::move(input);
		}

		void Update() override
		{
			_shift -= (_shiftGradient * learningRate + _previousShiftGradient * momentum) / minibatchSize;
			_scale -= (_scaleGradient * learningRate + _previousScaleGradient * momentum) / minibatchSize;

			_previousShiftGradient = _shiftGradient;
			_previousScaleGradient = _scaleGradient;

			_shiftGradient = 0.0f;
			_scaleGradient = 0.0f;
		}	

	private:	
		float _shift = 0.0f;
		float _scale = 1.0f;

		float _shiftGradient = 0.0f;
		float _scaleGradient = 0.0f;
		
		float _previousShiftGradient = 0.0f;
		float _previousScaleGradient = 0.0f;

		float _mean;
		float _variance;
};