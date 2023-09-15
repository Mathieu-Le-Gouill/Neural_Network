#pragma once

#include <iostream>

class MatrixF;
MatrixF dot(const MatrixF& a, const MatrixF& b);
void print(const MatrixF& matrix, float decimals = 2.f);
MatrixF rand(const uint16_t rows, const uint16_t cols, const float mean = 0.f, const float sigma = 1.f);
void fill(MatrixF& a, float value);

class MatrixF
{
public:

	MatrixF();
	MatrixF(const uint16_t rows, const uint16_t cols);
	MatrixF(const uint16_t rows, const uint16_t cols, float value);
	MatrixF(const uint16_t rows, const uint16_t cols, float* values);
	MatrixF(const MatrixF& other);

	~MatrixF();

	uint16_t rows() const;
	uint16_t cols() const;


	MatrixF operator+(const float value) const;
	MatrixF operator-(const float value) const;
	MatrixF operator*(const float value) const;
	MatrixF operator/(const float value) const;

	MatrixF operator+(const MatrixF& other) const;
	MatrixF operator-(const MatrixF& other) const;
	MatrixF operator*(const MatrixF& other) const;
	MatrixF operator/(const MatrixF& other) const;

	MatrixF& operator=(const MatrixF& other);

	void operator+=(const float value);
	void operator-=(const float value);
	void operator*=(const float value);
	void operator/=(const float value);

	void operator+=(const MatrixF& other);
	void operator-=(const MatrixF& other);
	void operator*=(const MatrixF& other);
	void operator/=(const MatrixF& other);

	float& operator()(const uint16_t row, const uint16_t col);
	float operator()(const uint16_t row, const uint16_t col) const;

	friend void add(const MatrixF& a, const float value, MatrixF& output);
	friend void sub(const MatrixF& a, const float value, MatrixF& output);
	friend void mul(const MatrixF& a, const float value, MatrixF& output);
	friend void div(const MatrixF& a, const float value, MatrixF& output);

	friend void add(const MatrixF& a, const MatrixF& b, MatrixF& output);
	friend void sub(const MatrixF& a, const MatrixF& b, MatrixF& output);
	friend void mul(const MatrixF& a, const MatrixF& b, MatrixF& output);
	friend void div(const MatrixF& a, const MatrixF& b, MatrixF& output);

	friend float sum(const MatrixF& matrix);

	friend float max(const MatrixF& matrix);

	friend uint32_t argmax(const MatrixF& matrix);

	friend MatrixF abs(const MatrixF& matrix);

	friend MatrixF exp(const MatrixF& matrix);

	friend void print(const MatrixF& matrix, float decimals);

	friend MatrixF relu(const MatrixF& matrix);
	friend MatrixF relu_derivative(const MatrixF& matrix);

	friend MatrixF leaky_relu(const MatrixF& matrix, const float slope);
	friend MatrixF leaky_relu_derivative(const MatrixF& matrix, const float slope);

	friend MatrixF sigmoid(const MatrixF& matrix);

	friend MatrixF softmax(const MatrixF& matrix);
	friend MatrixF softmax_derivative(const MatrixF& matrix);

	friend MatrixF transpose(const MatrixF& matrix);

	friend void swap(MatrixF& a, MatrixF& b);

	friend void move(MatrixF& a, MatrixF& b);

	friend MatrixF dot(const MatrixF& a, const MatrixF& b);

	friend float scalar(const MatrixF& a, const MatrixF& b);
	friend MatrixF scaled_sum(const MatrixF& a, const MatrixF& b);

	friend void fill(MatrixF& a, float value);

	friend MatrixF rand(const uint16_t rows, const uint16_t cols, const float mean, const float sigma);

private:
	uint16_t m_rows, m_cols;
	uint32_t m_length;

	uint32_t m_packageCount;
	uint32_t m_packageBack;

	float* m_begin;
	float* m_end;
	float* m_packageEnd;
};


