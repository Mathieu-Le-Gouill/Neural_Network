#pragma once
#include <cstdint>

inline constexpr float learningRate = 0.5f;
inline constexpr float momentum = 0.1f;

inline constexpr uint16_t minibatchSize = 20;
inline constexpr uint16_t epochs = 3000;
inline constexpr uint16_t nbTestImages = 10000;

inline constexpr uint16_t inputWidth = 28;
inline constexpr uint16_t inputHeight = 28;

inline constexpr uint16_t outputSize = 10;

static_assert(minibatchSize > 0, "batchSize must be greater than 0");
static_assert(epochs > 0, "epochs must be greater than 0");


inline constexpr uint16_t inputSize = inputWidth * inputHeight;

inline constexpr size_t nbTrainImages = epochs * minibatchSize;

static_assert(nbTrainImages <= 60000, "nbTrainImages must be inferior to 60 000");
static_assert(nbTestImages <= 10000, "nbTestImages must be inferior to 10 000");