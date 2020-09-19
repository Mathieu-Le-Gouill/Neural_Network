#pragma once
#include <vector>
#include "assert.h"
#include <iostream>
#include <array>

template <class Type>
class Matrix
{
public:
	Matrix();// Default constructor
	Matrix(unsigned nbCols, unsigned nbRows = 0, Type value = 0);// Overladen constructor
	Matrix(const std::vector<std::vector<Type> > &matrix);// Overladen constructor
	~Matrix();// Destructor

	void print(int approximateCoefficient = 10) const;// Method to show the matrix values 
	bool empty() const;// Method to check if the matrix is empty
	std::vector<Type>& back() const;// Method to obtain the last column of the matrix

	void add_a_Row(const std::vector<Type> &row);//Method to add a row to the matrix
	void add_a_Column(const std::vector<Type> &column);//Method to add a column to the matrix
	void setSize(unsigned nbCols, unsigned nbRows);// Method to set the matrix size

	unsigned size() const;// Method to get the matrix size : columns by rows
	unsigned rows() const;// Method to get the matrix rows number
	unsigned columns() const;// Method to get the matrix columns number
	Type determinant() const;// Method to get the matrix determinant
	Matrix<Type> identity() const;// Method to get the matrix identity

	Matrix<Type> transpose() const;// Method to get the matrix transpose : rows to columns
	Matrix<Type> inverse() const;// Method to get the matrix inverse 
	Matrix<Type> scalar(Matrix<Type> matrix) const;//Method to use the scalar product on a vector

	Matrix<Type> operator *(Type value) const;// Method to manipulate * operator with a constant value
	Matrix<Type> operator /(Type value) const;// Method to manipulate / operator with a constant value
	Matrix<Type> operator ^(Type value) const;// Method to manipulate ^ operator with a constant value

	Matrix<Type> operator +(const Matrix<Type> &matrix) const;// Method to manipulate + operator with a matrix
	Matrix<Type> operator -(const Matrix<Type> &matrix) const;// Method to manipulate - operator with a matrix
	Matrix<Type> operator *(const Matrix<Type> &matrix) const;// Method to manipulate * operator with a matrix
	Matrix<Type> operator /(const Matrix<Type> &matrix) const;// Method to manipulate / operator with a matrix
	Matrix<Type> operator =(const Matrix<Type> &matrix);// Method to manipulate = operator with a matrix

	void operator -=(Type value);// Method to manipulate -= operator with a constant value
	void operator +=(Type value);// Method to manipulate += operator with a constant value
	void operator *=(Type value);// Method to manipulate *= operator with a constant value
	void operator /=(Type value);// Method to manipulate /= operator with a constant value
	void operator ^=(Type value);// Method to manipulate ^= operator with a constant value

	void operator -=(const Matrix<Type> &matrix);// Method to manipulate -= operator with a matrix
	void operator +=(const Matrix<Type> &matrix);// Method to manipulate += operator with a matrix
	void operator *=(const Matrix<Type> &matrix);// Method to manipulate *= operator with a matrix
	void operator /=(const Matrix<Type> &matrix);// Method to manipulate /= operator with a matrix

	std::vector<Type>& operator [](unsigned index);// Method to manipulate [] operator with a matrix

	bool operator ==(const Matrix<Type> &matrix) const;//Method to manipulate == operator with a matrix
	bool operator !=(const Matrix<Type> &matrix) const;//Method to manipulate != operator with a matrix

private:

	unsigned m_nbRows;// Contains the matrix rows number 
	unsigned m_nbCols;// Contains the matrix columns number 

	std::vector<std::vector<Type> > m_values;// Contains all the matrix values
};


//-Methods Definitions

template<class Type>
Matrix<Type>::Matrix()// Default constructor
{
	m_nbRows = 0;
	m_nbCols = 0;
}


template<class Type>
Matrix<Type>::Matrix(unsigned nbCols, unsigned nbRows, Type value)// Overladen constructor 
{
	if (nbCols == 0 && nbRows != 0) nbCols = 1;
	else if (nbRows == 0 && nbCols != 0) nbRows = 1;

	this->m_nbRows = nbRows;
	this->m_nbCols = nbCols;

	for (unsigned r = 0; r < nbRows; r++)// For each rows
	{
		this->m_values.push_back(std::vector<Type>());// Add a new row

		for (unsigned c = 0; c < nbCols; c++)// For each columns
		{
			this->m_values[r].push_back(value);// Add a value to the matrix
		}
	}
}


template<class Type>
inline Matrix<Type>::Matrix(const std::vector<std::vector<Type>> &matrix)// Overladen constructor
{
	if (!matrix.empty())// If the given matrix is not empty
	{
		this->m_values = matrix;
		this->m_nbRows = matrix.size();
		this->m_nbCols = matrix.front().size();
	}
	else Matrix::Matrix();
}


template<class Type>
Matrix<Type>::~Matrix()// Destructor
{
}


template<class Type>
Matrix<Type> Matrix<Type>::operator *(Type value) const// Method to manipulate * operator
{
	Matrix<Type> matrix = *this;

	const unsigned &nbRows = matrix.m_nbRows;
	const unsigned &nbCols = matrix.m_nbCols;

	for (unsigned r = 0; r < nbRows; r++)// For each rows
	{
		for (unsigned c = 0; c < nbCols; c++)// For each columns
		{
			matrix.m_values[r][c] *= value;// Multiply all the matrix values by the given value
		}
	}
	return matrix;
}


template<class Type>
Matrix<Type> Matrix<Type>::operator /(Type value) const// Method to manipulate / operator with a constant value
{
	Matrix<Type> matrix = *this;

	const unsigned &nbRows = matrix.m_nbRows;
	const unsigned &nbCols = matrix.m_nbCols;

	for (unsigned r = 0; r < nbRows; r++)// For each rows
	{
		for (unsigned c = 0; c < nbCols; c++)// For each columns
		{
			matrix.m_values[r][c] /= value;// Divide all the matrix values by the given value
		}
	}
	return matrix;
}


template<class Type>
Matrix<Type> Matrix<Type>::operator ^(Type value) const// Method to manipulate ^ operator with a constant value
{
	assert(this->m_nbRows == this->m_nbCols && "Error : the given matrix must be square to use operator ^");// Be sure that this is a square matrix

	Matrix<Type> matrix = *this;

	for (int i = 1; i < value; i++)// From the current matrix power to the given value
	{
		matrix = matrix * matrix;// Multiply the matrix by itself to attein the power asked
	}
	if (value == -1)// If the given power is -1
	{
		matrix = matrix.inverse();// Inverse the matrix
	}

	return matrix;
}


template<class Type>
Matrix<Type> Matrix<Type>::operator *(const Matrix<Type> &matrix) const// Method to manipulate * operator with a matrix
{
	assert(this->m_nbCols == matrix.m_nbRows && "Error : the matrices used must have the relation, first matrix's columns number equal to second matrix's rows number to use operator * ...");
	// Be sure that the first matrix have the same number of columns than the second matrix rows number

	const Matrix<Type> &matrixA = *this;
	const Matrix<Type> &matrixB = matrix;
	Matrix<Type> matrixC;

	matrixC.m_nbRows = matrixA.m_nbRows;
	matrixC.m_nbCols = matrixB.m_nbCols;

	const unsigned &nbRows = matrixC.m_nbRows;
	const unsigned &nbCols = matrixC.m_nbCols;

	for (unsigned r = 0; r < nbRows; r++)// For each rows
	{
		matrixC.m_values.push_back(std::vector<Type>());
		for (unsigned c = 0; c < nbCols; c++)// For each columns
		{
			Type sum = 0;
			for (unsigned i = 0; i < matrixA.m_nbCols; i++)// For each columns of the matrix A or each matrix B rows
			{
				sum += matrixA.m_values[r][i] * matrixB.m_values[i][c];// Product of matrix A rows by matrix B columns
			}
			matrixC.m_values.back().push_back(sum);// Add the sum to matrix C value
		}
	}

	return matrixC;
}

template<class Type>
inline Matrix<Type> Matrix<Type>::operator/(const Matrix<Type>& matrix) const// Method to manipulate / operator with a matrix
{
	assert(this->m_nbCols == matrix.m_nbRows && matrix.m_nbRows == matrix.m_nbCols  && "Error :  the matrices used must have the relation, first matrix's columns number equal to second matrix's rows number and the matrix argument must be square to use operator / ...");
	// Be sure that the matrix argument is square and have as size the first matrix columns number 

	Matrix<Type> matrixA = *this;
	Matrix<Type> matrixB = matrix;

	return  matrixA * matrixB.inverse();
}


template<class Type>
Matrix<Type> Matrix<Type>::operator -(const Matrix<Type> &matrix) const// Method to manipulate - operator with a matrix
{
	assert(this->m_nbRows == matrix.m_nbRows && this->m_nbCols == matrix.m_nbCols && "Error : the matrices used must have an equal size to use the operator -");
	// Be sure that the both matrices have the same size

	Matrix<Type> matrixA = *this;
	const Matrix<Type> &matrixB = matrix;

	const unsigned &nbRows = this->m_nbRows;
	const unsigned &nbCols = this->m_nbCols;

	for (unsigned r = 0; r < nbRows; r++)// For each rows
	{
		for (unsigned c = 0; c < nbCols; c++)// For each columns
		{
			matrixA.m_values[r][c] -= matrixB.m_values[r][c];// Substract the matrix B values to the A
		}
	}
	return matrixA;
}


template<class Type>
Matrix<Type> Matrix<Type>::operator +(const Matrix<Type> &matrix) const// Method to manipulate + operator with a matrix
{
	assert(this->m_nbRows == matrix.m_nbRows && this->m_nbCols == matrix.m_nbCols && "Error : the matrices used must have an equal size to use the operator +");
	// Be sure that the both matrices have the same size

	Matrix<Type> matrixA = *this;
	const Matrix<Type> &matrixB = matrix;

	const unsigned &nbRows = matrix.m_nbRows;
	const unsigned &nbCols = matrix.m_nbCols;

	for (unsigned r = 0; r < nbRows; r++)// For each rows
	{
		for (unsigned c = 0; c < nbCols; c++)// For each columns
		{
			matrixA.m_values[r][c] += matrixB.m_values[r][c];// Add the matrix B values to the A
		}
	}
	return matrixA;
}


template<class Type>
 Matrix<Type> Matrix<Type>::operator =(const Matrix<Type> &matrix)// Method to manipulate = operator
{
	this->m_nbRows = matrix.m_nbRows;
	this->m_nbCols = matrix.m_nbCols;
	this->m_values = matrix.m_values;
	// Copy the matrix values;

	return *this;
}


 template<class Type>
 inline void Matrix<Type>::operator-=(Type value)// Method to manipulate -= operator with a constant value
 {
	 *this = *this - value;
 }


 template<class Type>
 inline void Matrix<Type>::operator+=(Type value)// Method to manipulate += operator with a constant value
 {
	 *this = *this + value;
 }


 template<class Type>
 inline void Matrix<Type>::operator*=(Type value)// Method to manipulate *= operator with a constant value
 {
	 *this = *this * value;
 }


 template<class Type>
 inline void Matrix<Type>::operator/=(Type value)// Method to manipulate /= operator with a constant value
 {
	 *this = *this / value;
 }

 template<class Type>
 inline void Matrix<Type>::operator^=(Type value)// Method to manipulate ^= operator with a constant value
 {
	 *this = *this ^ value;
 }


 template<class Type>
 inline void Matrix<Type>::operator -=(const Matrix<Type>& matrix)// Method to manipulate -= operator with a matrix
 {
	 *this = *this - matrix;
 }

 template<class Type>
 inline void Matrix<Type>::operator+=(const Matrix<Type>& matrix)// Method to manipulate += operator with a matrix
 {
	 *this = *this + matrix;
 }


 template<class Type>
 inline void Matrix<Type>::operator *=(const Matrix<Type>& matrix)// Method to manipulate *= operator with a matrix
 {
	 *this = *this * matrix;
 }

 template<class Type>
 inline void Matrix<Type>::operator/=(const Matrix<Type>& matrix)// Method to manipulate /= operator with a matrix
 {
	 *this = *this / matrix;
 }


 template<class Type>
 std::vector<Type>& Matrix<Type>::operator [](unsigned index)// Method to manipulate [] operator with a matrix
 {
	 assert(index < this->m_values.size() && "Error : the given value is out of the matrix range..."); 
	 // Be sure that the given index value belongs to the vector size

	 return this->m_values[index];
 }


 template<class Type>
 inline bool Matrix<Type>::operator ==(const Matrix<Type>& matrix) const//Method to manipulate == operator with a matrix
 {
	 return (this->m_values == matrix.m_values) ? true : false;
 }


 template<class Type>
 inline bool Matrix<Type>::operator !=(const Matrix<Type>& matrix) const//Method to manipulate != operator with a matrix
 {
	 return (this->m_values != matrix.m_values) ? true : false;
 }


template<class Type>
inline void Matrix<Type>::print(int approximateCoefficient) const// Method to show the matrix values 
{
	const unsigned &nbRows = this->m_nbRows;
	const unsigned &nbCols = this->m_nbCols;

	for (unsigned r = 0; r < nbRows; r++)// For each rows
	{
		for (unsigned c = 0; c < nbCols; c++)// For each columns
		{
			std::cout << round(this->m_values[r][c] * approximateCoefficient) / approximateCoefficient << " ";// Print the values according to the  to the approximate coefficient, 10 for exemple give the value to the nearest tenth 
		}
		std::cout <<"\n";
	}
}


template<class Type>
inline bool Matrix<Type>::empty() const// Method to check if the matrix is empty
{
	return (this->m_values.empty()) ?  true : false;// Return true if the matrix is empty else return false
}

template<class Type>
inline std::vector<Type>& Matrix<Type>::back() const// Method to obtain the last column of the matrix
{
	return this->m_values.back();// Return the last column of the matrix
}


template<class Type>
inline void Matrix<Type>::add_a_Row(const std::vector<Type> &row)//Method to add a row to the matrix
{
	if (this->empty()) this->m_nbCols = row.size();
	this->m_values.push_back(row);// Add the row to the matrix
	this->m_nbRows++;
}


template<class Type>
inline void Matrix<Type>::add_a_Column(const std::vector<Type> &column)//Method to add a column to the matrix
{
	const unsigned &nbRows = this->m_nbRows;
	if (this->empty()) this->setSize(1, column.size());

	for (unsigned r = 0; r < nbRows; r++)
		this->m_values[r].push_back(column[r]);// Add the column to the matrix
}


template<class Type>
inline void Matrix<Type>::setSize(unsigned nbCols, unsigned nbRows)// Method to set the matrix size
{
	if (this->m_nbRows != nbRows || this->m_nbCols != nbCols)
	{

		while (this->m_nbRows != nbRows)// While the matrix number rows still different that the number requested
		{
			if (this->m_nbRows < nbRows)// if the current matrix have not enought rows compared to the rows number requested
			{
				this->m_values.push_back(std::vector<Type>(this->m_nbCols, 0));// Add a row
				this->m_nbRows++;
			}

			else if (this->m_nbRows > nbRows)// if the current matrix have more rows than the rows number requested
			{
				this->m_values.pop_back(); // delete a row
				this->m_nbRows--;
			}

		}
		
		while (this->m_nbCols != nbCols)
		{
			if (this->m_nbCols < nbCols)// if the current matrix have not enought columns compared to the columns number requested
			{
				for (unsigned r = 0; r < nbRows; r++)
					this->m_values[r].push_back(0);// Add a column
				this->m_nbCols++;
			}

			else if (this->m_nbCols > nbCols)// if the current matrix have more columns than the columns number requested
			{
				for (unsigned r = 0; r < nbRows; r++)
					this->m_values[r].pop_back();// delete a column
				this->m_nbCols--;
			}
		}
		
	}
}

template<class Type>
inline unsigned Matrix<Type>::size() const// Method to get the matrix size
{
	return this->m_nbRows * this->m_nbCols;
}

template<class Type>
inline unsigned Matrix<Type>::rows() const// Method to ge the matrix rows number
{
	return this->m_nbRows;
}

template<class Type>
inline unsigned Matrix<Type>::columns() const// Method to ge the matrix columns number
{
	return this->m_nbCols;
}


template<class Type>
inline Type Matrix<Type>::determinant() const// Method to get the matrix determinant
{
	assert(this->m_nbRows == this->m_nbCols && "Error : the given matrix must be square to use determinant method...");
	// Be sure that this is a square matrix

	Type determinantValue = 0;

	const unsigned &nbRows = this->m_nbRows;
	const unsigned &nbCols = this->m_nbCols;

	const unsigned &matrixSize = nbRows;// Matrix parts size to compute the determinant

	if (matrixSize == 1)
	{
		determinantValue = this->m_values[0][0];
	}

	else if (matrixSize == 2)// In case of a 2*2 matrix
	{
		determinantValue = this->m_values[0][0] * this->m_values[1][1] - this->m_values[0][1] * this->m_values[1][0];
	}

	else if(matrixSize > 2)// In case of a matrix size greater than 2
	{
		for (unsigned i = 0; i < nbCols; i++)// For each values in the first line
		{
			Matrix matrix = *this;// Create a matrix part which will be smaller than the original

			for (unsigned r = 0; r < nbRows; r++) 
				matrix.m_values[r].erase(matrix.m_values[r].begin() + i);// Erase the i column from the matrix

			matrix.m_values.erase(matrix.m_values.begin());// Erase the first line of the original matrix

			matrix.m_nbRows = nbRows - 1;
			matrix.m_nbCols = nbCols - 1;

			determinantValue += this->m_values[0][i] * matrix.determinant() * pow((-1),i);// Compute the determinant
		}
	}

	return determinantValue;
}

template<class Type>
inline Matrix<Type> Matrix<Type>::identity() const// Method to get the matrix identity
{
	assert(this->m_nbRows == this->m_nbCols && "Error : the given matrix must be square to use identity method...");
	// Be sure that this is a square matrix

	Matrix<Type> matrix = *this;

	const unsigned &nbRows = this->m_nbRows;
	const unsigned &nbCols = this->m_nbCols;

	for (unsigned r = 0; r < nbRows; r++)// For each rows
	{
		for (unsigned c = 0; c < nbCols; c++)// For each columns
		{
			if (c == r)
				matrix.m_values[r][c] = 1;// Construct the middle diagonal of the identity matrix
			else
				matrix.m_values[r][c] = 0;// Fill the rest of the matrix
		}
		std::cout << "\n";
	}

	return matrix;
}


template<class Type>
Matrix<Type> Matrix<Type>::transpose() const// Method to transpose the matrix rows to collumns
{
	Matrix<Type> matrix;

	matrix.m_nbRows = this->m_nbCols;
	matrix.m_nbCols = this->m_nbRows;

	const unsigned &nbRows = matrix.m_nbRows;
	const unsigned &nbCols = matrix.m_nbCols;

	for (unsigned r = 0; r < nbRows; r++)// For each rows
	{
		matrix.m_values.push_back(std::vector<Type>());
		for (unsigned c = 0; c < nbCols; c++)// For each columns
		{
			matrix.m_values.back().push_back(this->m_values[c][r]);// Transpose the matrix rows to columns
		}
	}
	return matrix;
}

template<class Type>
inline Matrix<Type> Matrix<Type>::inverse() const// Method to inverse the matrix
{
	assert(this->m_nbRows == this->m_nbCols && "Error : the given matrix must be square to use inverse method...");
	// Be sure that this is a square matrix

	assert(this->determinant() != 0 && "Error : the given matrix can't be inversed, the matrix determinant is equal to 0...");
	// Be sure that the determinant of the given matrix is different to 0

	Matrix<Type> matrix = *this;

	const unsigned &matrixSize = this->m_nbCols;// Matrix square size


	if (matrixSize == 2)// In case of a 2*2 matrix
	{
		matrix.m_values[0][0] = this->m_values[1][1];  matrix.m_values[0][1] = -this->m_values[0][1];
		matrix.m_values[1][0] = -this->m_values[1][0]; matrix.m_values[1][1] =  this->m_values[0][0];
	}

	else if (matrixSize > 2)// In case of a matrix size greater than 2*2
	{
		const unsigned &nbRows = this->m_nbRows;
		const unsigned &nbCols = this->m_nbCols;

		for (unsigned r = 0; r < nbRows; r++)// For each rows in the matrix
		{
			for (unsigned c = 0; c < nbCols; c++)// For each columns in the matrix
			{
				Matrix matrix_part = *this;// Create a matrix part which will be smaller than the original

				for (unsigned r = 0; r < nbRows; r++)
					matrix_part.m_values[r].erase(matrix_part.m_values[r].begin() + c);// Erase the column from the matrix

				matrix_part.m_values.erase(matrix_part.m_values.begin() + r);// Erase the row from the matrix

				matrix_part.m_nbRows = nbRows - 1;
				matrix_part.m_nbCols = nbCols - 1;

				matrix.m_values[c][r] = matrix_part.determinant() * (double)pow((-1), r + c);// Compute inverse matrix values
			}
		}
	}

	matrix = matrix * (1 / this->determinant());// Compute the final inverse

	return matrix;
}


template<class Type>
inline Matrix<Type> Matrix<Type>::scalar(Matrix<Type> matrix) const
{
	const Matrix<Type> &matrixA = *this;
	const Matrix<Type> &matrixB = matrix;
	assert(matrixA.rows() == matrixB.rows() && matrixA.columns() == matrixB.columns() && (matrixA.rows() == 1 || matrixA.columns() == 1) && "Error : the matrices used must have an equal size to use the scalar function");
	// Be sure that the both matrices have the same size and are both vectors

	Matrix<Type> matrixC = matrixA;
	unsigned nbRows = matrixC.rows();
	unsigned nbCols = matrixC.columns();

	for (unsigned r = 0; r < nbRows; r++)// For each rows
	{
		for (unsigned c = 0; c < nbCols; c++)// For each columns
		{
			matrixC.m_values[r][c] *= matrixB.m_values[r][c];// Sclalar product of matrixA and matrixB
		}
	}

	return matrixC;
}

