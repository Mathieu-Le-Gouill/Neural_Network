#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "debug.h"
#include "MatrixF.h"

using namespace std;


int ReverseInt(int x);

	MatrixF* Load_MNIST_File(const string &MNIST_FilePath, unsigned nbImages)// Function to obtain the data inputs of the images from the MNIST training file
	{
		debug_assert(nbImages <= 60000);

		MatrixF* inputsValues = new MatrixF[nbImages];

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
				inputsValues[i] = MatrixF(1, n_rows*n_cols);

				for (int r = 0; r < n_rows; ++r)
				{
					
					for (int c = 0; c < n_cols; ++c)
					{
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));
						inputsValues[i](0, r*n_cols+c) = (float)temp / 255.f;
					}
					
				}
			}
		}
		else
			cout << "Error loading MNIST_File..." << endl;

		return inputsValues;
	}


	MatrixF* GetTargetValues(const string &LabelFilePath, unsigned nbImages)// Function to obtain the desired output for each images
	{
		MatrixF* targetsValues = new MatrixF[nbImages];
		ifstream file(LabelFilePath.c_str(), ios::binary);
		debug_assert(nbImages <= 60000);

		if (file.is_open())
		{
			int magic_number = 0;
			int number_of_images = 0;

			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);

			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);

			for (unsigned i = 0; i < nbImages; ++i)
			{
				targetsValues[i] = MatrixF(1, 10, 0.f);

				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));

				targetsValues[i](0, (uint16_t)temp) = 1.0;
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
