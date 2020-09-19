#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include "Matrix.h"

using namespace std;


int ReverseInt(int x);

	vector<Matrix<double>> Load_MNIST_File(const string &MNIST_FilePath, int nbImages, int ImageDataSize)
	//	:m_nbTrainingImages(NumberOfImages)
	{
		//vector<vector<double> > inputsValues;
		vector<Matrix<double>> inputsValues;
		assert(ImageDataSize == 784);
		assert(nbImages <= 60000);
		//inputsValues.resize(nbImages, vector<double>(ImageDataSize));

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

			for (int i = 0; i < nbImages; ++i)
			{
				inputsValues.push_back(Matrix<double >());
				vector<double> values;

				for (int r = 0; r < n_rows; ++r)
				{
					
					for (int c = 0; c < n_cols; ++c)
					{
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));
						values.push_back((double)temp / 255.0);
					}
					
				}
				inputsValues.back().add_a_Row(values);
			}
		}
		else
			cout << "Error loading MNIST_File..." << endl;

		return inputsValues;
	}


	vector<Matrix<double>> GetTargetValues(const string &LabelFilePath, int nbImages)
	{
		vector<double> target;
		vector<Matrix<double>> targetsValues;
		//targetsValues.resize(nbImages, vector<double>(10));
		ifstream file(LabelFilePath.c_str(), ios::binary);
		assert(nbImages <= 60000);

		if (file.is_open())
		{
			int magic_number = 0;
			int number_of_images = 0;

			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);

			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);

			for (int i = 0; i < nbImages; ++i)
			{
				targetsValues.push_back(Matrix<double>());
				vector<double> values;

				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				target.push_back((double)temp);

				for (int a = 0; a < 10; a++)
				{
					if (a == target.back())
						values.push_back(1);
					//targetsValues[i][a] = 1;
					else
						values.push_back(0);
						//targetsValues[i][a] = 0;
				}
				targetsValues.back().add_a_Row(values);
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