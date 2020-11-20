#pragma once
#include <vector>

class Matrix
{
public :
	Matrix();
	~Matrix();

	Matrix(const Matrix& matrix);
	Matrix(const std::vector<std::vector<double>>& val);
	Matrix(unsigned nbRows, unsigned nbCols, const double* val);
	Matrix(unsigned nbRows, unsigned nbCols, double value = 0.0);

	void print(int approximationCoefficient = 10)  const;
	bool empty() const;
	unsigned rows() const;
	unsigned cols() const;

	Matrix transpose() const;
	Matrix scalar(const Matrix& matrix) const;

	double& operator ()(unsigned row, unsigned column);

	Matrix operator *(double value) const;
	Matrix operator /(double value) const;
	Matrix operator ^(double value) const;

	Matrix operator +(const Matrix& matrix) const;
	Matrix operator -(const Matrix& matrix) const;
	Matrix operator *(const Matrix& matrix) const;

	Matrix& operator =(const Matrix& matrix);

	void operator *=(double value);
	void operator /=(double value);
	void operator ^=(double value);

	void operator +=(const Matrix& matrix);
	void operator -=(const Matrix& matrix);
	void operator *=(const Matrix& matrix);

private:
	double** m_values;
	unsigned m_nbCols;
	unsigned m_nbRows;
};

