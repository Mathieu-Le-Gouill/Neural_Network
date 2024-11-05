#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include "debug.h"
#include "Tensor.h"
#include "network_parameters.h"
#include <array>

using namespace std;

static int ReverseInt(int x);


// Function to obtain the data inputs of the images from the MNIST training file
static Tensor<inputWidth, inputHeight>* Load_MNIST_File(const string& MNIST_FilePath, const uint16_t nbImages)
{
	Tensor<inputWidth, inputHeight>* inputsValues = new Tensor<inputWidth, inputHeight>[nbImages];

	ifstream file(MNIST_FilePath.c_str(), ios::binary);

	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);

		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);

		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		for (unsigned i = 0; i < nbImages; ++i)
		{
			float data[inputWidth * inputHeight];

			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));

					data[r * n_cols + c] = (float)temp / 255.f;
				}
			}
			inputsValues[i] = Tensor<inputWidth, inputHeight>(data);			
		}
	}
	else
		cout << "Error loading MNIST_File..." << endl;

	return inputsValues;
}


// Function to obtain the desired output for each images
static Tensor<outputSize>* GetTargetValues(const string& LabelFilePath, const uint16_t nbImages)
{
	Tensor<outputSize>* targetsValues = new Tensor<outputSize>[nbImages];
	ifstream file(LabelFilePath.c_str(), ios::binary);


	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);

		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		for (size_t i = 0; i < nbImages; ++i)
		{
			float data[outputSize] = {0};

			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));

			data[(uint16_t)temp] = 1.0;

			targetsValues[i] = Tensor<outputSize>(data);
		}
	}
	else
		cout << "Error loading Label File..." << endl;

	return targetsValues;
}



int ReverseInt(int x)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = x & 255;
	ch2 = (x >> 8) & 255;
	ch3 = (x >> 16) & 255;
	ch4 = (x >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}