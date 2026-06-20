#pragma once
#include <cassert>
#include <complex>
#include <iomanip>
#include <ostream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <vector>
#include "../io/LibBSE_io.hpp"
#include "../interface/Blas_Interface.h"

namespace LibBSE{
    //Matrix: only for 2 dimension case.
    template <typename T>
    class matrix{
    public:
        int row;
        int col;
        int size;
        T *matrix_ptr;
        matrix(): 
        row(0), col(0), size(0), matrix_ptr(nullptr){
        }
	    matrix( const int nrows, const int ncols): 
        row(nrows), col(ncols), size(0), matrix_ptr(nullptr){
            if(row && col){
                size = row * col;
	        	matrix_ptr = new T[size];
	        }
        }
	    matrix( const matrix &m ): 
        row(m.row), col(m.col), size(m.size), matrix_ptr(nullptr){
            if(size){
                matrix_ptr = new T[size];
                for(int i = 0; i < size; ++i){
                    matrix_ptr[i] = m.matrix_ptr[i];
                }
            }
        }
	    matrix( matrix && m ):
        row(m.row), col(m.col), size(m.size), matrix_ptr(m.matrix_ptr){
            m.row = m.col = m.size = 0;
            m.matrix_ptr = nullptr;
        }
        
	    ~matrix(){
            if(matrix_ptr){
	        	delete[] matrix_ptr;
	        	matrix_ptr = nullptr;
	        }
        }
        //operators
	    matrix& operator=(const matrix &m); 
	    matrix& operator=( matrix && m );	
	    T &operator()(const int r,const int c);
	    const T &operator() (const int r,const int c) const;
	    void operator*=(const T &s);
	    void operator+=(const T &s);
	    void operator+=(const matrix &m);
	    void operator-=(const T &s);
	    void operator-=(const matrix &m);
	    bool operator==(const matrix &m);
        bool is_empty();
        void empty();
    };
    
    template<typename T>
    struct is_blas_matrix_value : std::false_type {};
    template<>
    struct is_blas_matrix_value<float> : std::true_type {};
    template<>
    struct is_blas_matrix_value<double> : std::true_type {};
    template<>
    struct is_blas_matrix_value<std::complex<float>> : std::true_type {};
    template<>
    struct is_blas_matrix_value<std::complex<double>> : std::true_type {};

    template<typename T>
    matrix<T> operator+(const matrix<T> &m1, const matrix<T> &m2);
    template<typename T>
    matrix<T> operator-(const matrix<T> &m1, const matrix<T> &m2);
    template<typename T>
    matrix<T> operator*(const matrix<T> &m1, const matrix<T> &m2);
    template<typename T>
    matrix<T> operator*(const T &s, const matrix<T> &m);
    template<typename T>
    matrix<T> operator*(const matrix<T> &m, const T &s);
    //use LibBSE_print
    template<typename T>
    void print_matrix(LibBSE::MpiComm MPI_COMM, const char *name, const matrix<T> &mat);
    //complex operater
    template<typename T>
    matrix<T>  transpose_times(const matrix<T> &m1, const matrix<T> &m2, const char T1, const char T2);
    template<typename T>
    void zeros(matrix<T>& mat);

    // Copy assignment: keep the matrix shape and data exactly the same as m.
    // The old buffer is reused only when the element count is unchanged.
    template<typename T>
    matrix<T>& matrix<T>::operator=(const matrix &m){
        if(this == &m){
            return *this;
        }

        // Reallocate only when the storage size changes.
        // This keeps assignment cheap for matrices with the same shape.
        if(size != m.size){
            if(matrix_ptr){
                delete[] matrix_ptr;
                matrix_ptr = nullptr;
            }
            if(m.size){
                matrix_ptr = new T[m.size];
            }
        }

        row = m.row;
        col = m.col;
        size = m.size;

        // Data is stored in row-major order:
        // index = row_index * number_of_columns + column_index.
        for(int i = 0; i < size; ++i){
            matrix_ptr[i] = m.matrix_ptr[i];
        }
        return *this;
    }

    // Move assignment: release this object's buffer, then take ownership of m.
    // m is reset to an empty matrix so its destructor is safe.
    template<typename T>
    matrix<T>& matrix<T>::operator=(matrix &&m){
        if(this == &m){
            return *this;
        }

        if(matrix_ptr){
            delete[] matrix_ptr;
        }

        row = m.row;
        col = m.col;
        size = m.size;
        matrix_ptr = m.matrix_ptr;

        m.row = 0;
        m.col = 0;
        m.size = 0;
        m.matrix_ptr = nullptr;

        return *this;
    }

    // Element access.  Bounds are checked in debug builds by assert.
    template<typename T>
    T& matrix<T>::operator()(const int r, const int c){
        assert(r >= 0 && r < row);
        assert(c >= 0 && c < col);
        return matrix_ptr[r * col + c];
    }

    // Const element access for read-only matrices.
    template<typename T>
    const T& matrix<T>::operator()(const int r, const int c) const{
        assert(r >= 0 && r < row);
        assert(c >= 0 && c < col);
        return matrix_ptr[r * col + c];
    }

    // In-place scalar multiplication.
    template<typename T>
    void matrix<T>::operator*=(const T &s){
        for(int i = 0; i < size; ++i){
            matrix_ptr[i] *= s;
        }
    }

    // In-place scalar addition.
    template<typename T>
    void matrix<T>::operator+=(const T &s){
        for(int i = 0; i < size; ++i){
            matrix_ptr[i] += s;
        }
    }

    // In-place matrix addition.  Both matrices must have the same shape.
    template<typename T>
    void matrix<T>::operator+=(const matrix &m){
        assert(row == m.row);
        assert(col == m.col);
        for(int i = 0; i < size; ++i){
            matrix_ptr[i] += m.matrix_ptr[i];
        }
    }

    // In-place scalar subtraction.
    template<typename T>
    void matrix<T>::operator-=(const T &s){
        for(int i = 0; i < size; ++i){
            matrix_ptr[i] -= s;
        }
    }

    // In-place matrix subtraction.  Both matrices must have the same shape.
    template<typename T>
    void matrix<T>::operator-=(const matrix &m){
        assert(row == m.row);
        assert(col == m.col);
        for(int i = 0; i < size; ++i){
            matrix_ptr[i] -= m.matrix_ptr[i];
        }
    }

    // Matrix equality: shapes must match first, then every element is checked.
    template<typename T>
    bool matrix<T>::operator==(const matrix &m){
        if(row != m.row || col != m.col || size != m.size){
            return false;
        }
        for(int i = 0; i < size; ++i){
            if(!(matrix_ptr[i] == m.matrix_ptr[i])){
                return false;
            }
        }
        return true;
    }

    template<typename T>
    bool matrix<T>::is_empty(){
        if(row == 0 || col == 0 || size == 0){
            return true;
        }
        return false;
    }

    template<typename T>
    void matrix<T>::empty(){
        delete[] matrix_ptr;
        row = 0;
        col = 0;
        size = 0;
        matrix_ptr = nullptr;
    }


    // Matrix + matrix.  A new matrix is returned; inputs are not changed.
    template<typename T>
    matrix<T> operator+(const matrix<T> &m1, const matrix<T> &m2){
        assert(m1.row == m2.row);
        assert(m1.col == m2.col);

        matrix<T> result(m1.row, m1.col);
        for(int i = 0; i < result.size; ++i){
            result.matrix_ptr[i] = m1.matrix_ptr[i] + m2.matrix_ptr[i];
        }
        return result;
    }

    // Matrix - matrix.  A new matrix is returned; inputs are not changed.
    template<typename T>
    matrix<T> operator-(const matrix<T> &m1, const matrix<T> &m2){
        assert(m1.row == m2.row);
        assert(m1.col == m2.col);

        matrix<T> result(m1.row, m1.col);
        for(int i = 0; i < result.size; ++i){
            result.matrix_ptr[i] = m1.matrix_ptr[i] - m2.matrix_ptr[i];
        }
        return result;
    }

    // Matrix multiplication.
    // m1 is (row x common), m2 is (common x col), result is (row x col).
    template<typename T>
    matrix<T> operator*(const matrix<T> &m1, const matrix<T> &m2){
        assert(m1.col == m2.row);

        matrix<T> result(m1.row, m2.col);
        if(result.size == 0){
            return result;
        }
        if(m1.col == 0){
            for(int i = 0; i < result.size; ++i){
                result.matrix_ptr[i] = T();
            }
            return result;
        }

        if constexpr (is_blas_matrix_value<T>::value){
            Blas_Interface::gemm('N', 'N',
                                 m1.row, m2.col, m1.col,
                                 T(1), m1.matrix_ptr, m1.col,
                                 m2.matrix_ptr, m2.col,
                                 T(0), result.matrix_ptr, result.col);
        }
        else{
            for(int i = 0; i < m1.row; ++i){
                for(int j = 0; j < m2.col; ++j){
                    T sum = T();
                    for(int k = 0; k < m1.col; ++k){
                        sum += m1(i, k) * m2(k, j);
                    }
                    result(i, j) = sum;
                }
            }
        }
        return result;
    }

    // Scalar * matrix.
    template<typename T>
    matrix<T> operator*(const T &s, const matrix<T> &m){
        matrix<T> result(m.row, m.col);
        for(int i = 0; i < result.size; ++i){
            result.matrix_ptr[i] = s * m.matrix_ptr[i];
        }
        return result;
    }

    // Matrix * scalar.
    template<typename T>
    matrix<T> operator*(const matrix<T> &m, const T &s){
        matrix<T> result(m.row, m.col);
        for(int i = 0; i < result.size; ++i){
            result.matrix_ptr[i] = m.matrix_ptr[i] * s;
        }
        return result;
    }

    template<typename T>
    void zeros(matrix<T>& mat){
        for(int i = 0; i < mat.size; ++i){
            mat.matrix_ptr[i] = 0.0;
        }
    }

    // Print matrix on root rank only.  ostream keeps this function generic for
    // int, double, complex<double>, and other types with operator<<.
    template<typename T>
    void print_matrix(LibBSE::MpiComm MPI_COMM, const char *name, const matrix<T> &mat){
        std::ostringstream os;

        os << name << " = [\n";
        for(int i = 0; i < mat.row; ++i){
            os << "  ";
            for(int j = 0; j < mat.col; ++j){
                os << std::setw(16) << std::fixed << std::setprecision(10) << mat(i, j);
            }
            os << "\n";
        }
        os << "]\n";

        LibBSE_printf_root(MPI_COMM, "%s", os.str().c_str());
    }

    template<typename T>
    matrix<T> transpose_times(const matrix<T> &m1, const matrix<T> &m2, const char T1, const char T2){
        assert((T1 == 'N' || T1 == 'T') && (T2 == 'N' || T2 == 'T'));

        int m  = (T1 == 'T') ? m1.col : m1.row;
        int k1 = (T1 == 'T') ? m1.row : m1.col;
        int k2 = (T2 == 'T') ? m2.col : m2.row;
        int n  = (T2 == 'T') ? m2.row : m2.col;

        assert(k1 == k2);
        int k = k1;

        matrix<T> result(m, n);
        if (m == 0 || n == 0 || k == 0) return result;

        if constexpr (is_blas_matrix_value<T>::value) {
            Blas_Interface::gemm(T1, T2, m, n, k, 
                                 T(1), m1.matrix_ptr, m1.col, 
                                       m2.matrix_ptr, m2.col, 
                                 T(0), result.matrix_ptr, result.col);
        } 
        else {
            auto get_m1 = [&](int r, int c) { return (T1 == 'T') ? m1(c, r) : m1(r, c); };
            auto get_m2 = [&](int r, int c) { return (T2 == 'T') ? m2(c, r) : m2(r, c); };

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    T sum = 0;
                    for (int l = 0; l < k; ++l) {
                        sum += get_m1(i, l) * get_m2(l, j);
                    }
                    result(i, j) = sum; 
                }
            }
        }

        return result;
    }       
}
