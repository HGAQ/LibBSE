#pragma once
#include <atomic>
#include <cassert>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <vector>
#include "../io/LibBSE_io.hpp"
#include "../interface/Blas_Interface.h"

namespace LibBSE{
    typedef std::complex<double> complex;

    struct MatrixMemoryTracker{
        inline static std::atomic<long long> current_bytes{0};
        inline static std::atomic<long long> peak_bytes{0};
        inline static std::atomic<long long> live_buffers{0};
        inline static std::atomic<long long> total_allocations{0};

        static void add_bytes(const long long bytes){
            if(bytes <= 0){
                return;
            }
            const long long current =
                current_bytes.fetch_add(bytes) + bytes;
            live_buffers.fetch_add(1);
            total_allocations.fetch_add(1);

            long long peak = peak_bytes.load();
            while(current > peak
               && !peak_bytes.compare_exchange_weak(peak, current)){
            }
        }

        static void remove_bytes(const long long bytes){
            if(bytes <= 0){
                return;
            }
            current_bytes.fetch_sub(bytes);
            live_buffers.fetch_sub(1);
        }
    };

    inline double matrix_bytes_to_mib(const long long bytes){
        return static_cast<double>(bytes) / 1024.0 / 1024.0;
    }

    inline void report_matrix_memory(const LibBSE::MpiComm& Comm,
                                     const char* label = "matrix memory"){
        const long long local_current =
            MatrixMemoryTracker::current_bytes.load();
        const long long local_peak =
            MatrixMemoryTracker::peak_bytes.load();
        const long long local_live =
            MatrixMemoryTracker::live_buffers.load();
        const long long local_alloc =
            MatrixMemoryTracker::total_allocations.load();

        long long sum_current = 0;
        long long max_current = 0;
        long long max_peak = 0;
        long long sum_live = 0;
        long long sum_alloc = 0;

        MPI_Reduce(&local_current, &sum_current, 1, MPI_LONG_LONG,
                   MPI_SUM, 0, Comm.LibBSE_MPI_raw());
        MPI_Reduce(&local_current, &max_current, 1, MPI_LONG_LONG,
                   MPI_MAX, 0, Comm.LibBSE_MPI_raw());
        MPI_Reduce(&local_peak, &max_peak, 1, MPI_LONG_LONG,
                   MPI_MAX, 0, Comm.LibBSE_MPI_raw());
        MPI_Reduce(&local_live, &sum_live, 1, MPI_LONG_LONG,
                   MPI_SUM, 0, Comm.LibBSE_MPI_raw());
        MPI_Reduce(&local_alloc, &sum_alloc, 1, MPI_LONG_LONG,
                   MPI_SUM, 0, Comm.LibBSE_MPI_raw());

        LibBSE_printf_root(Comm,
            "---------------------------------------------------\n"
            "Matrix Memory usage [%s]:\n"
            "---------------------------------------------------\n"
            "\tcurrent_sum\t=\t%.3f MiB \n"
            "\tcurr_max_rank\t=\t%.3f MiB\n\tpeak_max_rank\t=\t%.3f MiB \n"
            //"\tlive_buffers=%lld\n\ttotal_allocations=%lld\n"
            "---------------------------------------------------\n"
            ,
            label,
            matrix_bytes_to_mib(sum_current),
            matrix_bytes_to_mib(max_current),
            matrix_bytes_to_mib(max_peak));
    }

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
                MatrixMemoryTracker::add_bytes(
                    static_cast<long long>(size) * sizeof(T));
	        }
        }
	    matrix( const matrix &m ): 
        row(m.row), col(m.col), size(m.size), matrix_ptr(nullptr){
            if(size){
                matrix_ptr = new T[size];
                MatrixMemoryTracker::add_bytes(
                    static_cast<long long>(size) * sizeof(T));
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
                MatrixMemoryTracker::remove_bytes(
                    static_cast<long long>(size) * sizeof(T));
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
        bool reshape(const int r, const int c);
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
    matrix<T>  pointer_times(T* &m1, const matrix<T> &m2, const char T1, const char T2, int col1, int row1);

    template<typename T>
    void zeros(matrix<T>& mat);
    template<typename T>
    std::vector<T*>  split(const matrix<T> &m1, std::vector<int> idx);

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
                MatrixMemoryTracker::remove_bytes(
                    static_cast<long long>(size) * sizeof(T));
                delete[] matrix_ptr;
                matrix_ptr = nullptr;
            }
            if(m.size){
                matrix_ptr = new T[m.size];
                MatrixMemoryTracker::add_bytes(
                    static_cast<long long>(m.size) * sizeof(T));
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
            MatrixMemoryTracker::remove_bytes(
                static_cast<long long>(size) * sizeof(T));
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

    // Reshape only changes the visible row/column shape.
    // The storage is already one-dimensional, so matrix_ptr is untouched.
    template<typename T>
    bool matrix<T>::reshape(const int r, const int c){
        if(r < 0 || c < 0 || r * c != size){
            return false;
        }
        row = r;
        col = c;
        return true;
    }

    template<typename T>
    void matrix<T>::empty(){
        if(matrix_ptr){
            MatrixMemoryTracker::remove_bytes(
                static_cast<long long>(size) * sizeof(T));
            delete[] matrix_ptr;
        }
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

    template<typename T>
    matrix<T> pointer_times(T* &m1, const matrix<T> &m2, const char T1, const char T2, int col1, int row1){
        assert((T1 == 'N' || T1 == 'T') && (T2 == 'N' || T2 == 'T'));
        assert(row1 >= 0 && col1 >= 0);
        assert(m1 != nullptr || row1 * col1 == 0);

        // m1 is a row-major matrix view with shape row1 x col1.
        // Only the shape is supplied here; pointer_times does not own m1.
        int m  = (T1 == 'T') ? col1 : row1;
        int k1 = (T1 == 'T') ? row1 : col1;
        int k2 = (T2 == 'T') ? m2.col : m2.row;
        int n  = (T2 == 'T') ? m2.row : m2.col;

        assert(k1 == k2);
        int k = k1;

        matrix<T> result(m, n);
        if(m == 0 || n == 0 || k == 0){
            return result;
        }

        if constexpr (is_blas_matrix_value<T>::value) {
            // Blas_Interface::gemm accepts row-major leading dimensions.
            // For m1 the leading dimension is the original pointer-view col1.
            Blas_Interface::gemm(T1, T2, m, n, k,
                                 T(1), m1, col1,
                                       m2.matrix_ptr, m2.col,
                                 T(0), result.matrix_ptr, result.col);
        }
        else {
            auto get_m1 = [&](int r, int c) {
                return (T1 == 'T') ? m1[c * col1 + r] : m1[r * col1 + c];
            };
            auto get_m2 = [&](int r, int c) {
                return (T2 == 'T') ? m2(c, r) : m2(r, c);
            };

            for(int i = 0; i < m; ++i){
                for(int j = 0; j < n; ++j){
                    T sum = 0;
                    for(int l = 0; l < k; ++l){
                        sum += get_m1(i, l) * get_m2(l, j);
                    }
                    result(i, j) = sum;
                }
            }
        }

        return result;
    }

    // Try to split the vector to various parts, base on first_index, didn't return matrix vector 
    // but the matrix ptr of the first index
    // e.g. :
    // m1: 12*12, idx: {0, 2, 10}
    // return: vector{ *[0][0], *[2][0], *[10][0]}
    template<typename T>
    std::vector<T*>  split(const matrix<T> &m1, std::vector<int> idx){
        std::vector<T*> ptrs;
        ptrs.reserve(idx.size());
        for(int first_row : idx){
            assert(first_row >= 0);
            assert(first_row < m1.row);
            ptrs.emplace_back(m1.matrix_ptr + first_row * m1.col);
        }
        return ptrs;
    }
}
