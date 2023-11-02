#pragma once
#include <cstdint>

constexpr float learningRate = 0.01f;
constexpr float momentum = 0.9f;
constexpr uint16_t batchSize = 10;
constexpr uint16_t epochs = 10;

constexpr uint16_t inputWidth = 28;
constexpr uint16_t inputHeight = 28;
constexpr uint16_t inputChannels = 1;

constexpr uint16_t inputSize = inputWidth * inputHeight * inputChannels;

constexpr uint16_t outputSize = 10;