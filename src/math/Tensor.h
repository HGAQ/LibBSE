#pragma once
#include <cassert>
#include <iomanip>
#include <initializer_list>
#include <ostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include "../io/LibBSE_io.hpp"

namespace LibBSE{
    //Tensor: simple dense tensor storage for small LibBSE data blocks.
    //Data is stored in row-major order over shape[0], shape[1], ..., shape[dim-1].
    template <typename T>
    class tensor{
    public:
        int dim;
        std::vector<int> shape;
        int size;
        T *tensor_ptr;

        tensor():
        dim(0), shape(), size(0), tensor_ptr(nullptr){
        }

        tensor(const std::vector<int> &dims):
        dim(static_cast<int>(dims.size())), shape(dims), size(1), tensor_ptr(nullptr){
            for(int i = 0; i < dim; ++i){
                if(shape[static_cast<std::size_t>(i)] <= 0){
                    size = 0;
                    break;
                }
                size *= shape[static_cast<std::size_t>(i)];
            }
            if(size){
                tensor_ptr = new T[size];
            }
        }

        tensor(const std::initializer_list<int> &dims):
        tensor(std::vector<int>(dims)){
        }

        tensor(const int n1, const int n2, const int n3):
        tensor(std::vector<int>{n1, n2, n3}){
        }

        tensor(const tensor &t):
        dim(t.dim), shape(t.shape), size(t.size), tensor_ptr(nullptr){
            if(size){
                tensor_ptr = new T[size];
                for(int i = 0; i < size; ++i){
                    tensor_ptr[i] = t.tensor_ptr[i];
                }
            }
        }

        tensor(tensor &&t):
        dim(t.dim), shape(std::move(t.shape)), size(t.size), tensor_ptr(t.tensor_ptr){
            t.dim = 0;
            t.size = 0;
            t.tensor_ptr = nullptr;
        }

        ~tensor(){
            if(tensor_ptr){
                delete[] tensor_ptr;
                tensor_ptr = nullptr;
            }
        }

        //operators
        tensor& operator=(const tensor &t);
        tensor& operator=(tensor &&t);
        T &operator()(const std::vector<int> &index);
        const T &operator()(const std::vector<int> &index) const;
        T &operator()(const int i, const int j, const int k);
        const T &operator()(const int i, const int j, const int k) const;
        void operator*=(const T &s);
        void operator+=(const T &s);
        void operator+=(const tensor &t);
        void operator-=(const T &s);
        void operator-=(const tensor &t);
        bool operator==(const tensor &t);

    private:
        int offset(const std::vector<int> &index) const;
    };

    template<typename T>
    tensor<T> operator+(const tensor<T> &t1, const tensor<T> &t2);
    template<typename T>
    tensor<T> operator-(const tensor<T> &t1, const tensor<T> &t2);
    template<typename T>
    tensor<T> operator*(const tensor<T> &t1, const tensor<T> &t2);
    template<typename T>
    tensor<T> operator*(const T &s, const tensor<T> &t);
    template<typename T>
    tensor<T> operator*(const tensor<T> &t, const T &s);
    //use LibBSE_print
    template<typename T>
    void print_tensor(LibBSE::MpiComm MPI_COMM, const char *name, const tensor<T> &ten);

        // Convert a multi-dimensional index to the flat row-major position.
    // Example for shape {a,b,c}: index {i,j,k} -> (i*b + j)*c + k.
    template<typename T>
    int tensor<T>::offset(const std::vector<int> &index) const{
        assert(static_cast<int>(index.size()) == dim);

        int pos = 0;
        for(int i = 0; i < dim; ++i){
            assert(index[static_cast<std::size_t>(i)] >= 0);
            assert(index[static_cast<std::size_t>(i)] < shape[static_cast<std::size_t>(i)]);
            pos = pos * shape[static_cast<std::size_t>(i)]
                + index[static_cast<std::size_t>(i)];
        }
        return pos;
    }

    // Copy assignment: copy both the tensor shape and all stored values.
    // The old buffer is reused when the flat size is unchanged.
    template<typename T>
    tensor<T>& tensor<T>::operator=(const tensor &t){
        if(this == &t){
            return *this;
        }

        // Reallocate only when the number of elements changes.
        if(size != t.size){
            if(tensor_ptr){
                delete[] tensor_ptr;
                tensor_ptr = nullptr;
            }
            if(t.size){
                tensor_ptr = new T[t.size];
            }
        }

        dim = t.dim;
        shape = t.shape;
        size = t.size;

        for(int i = 0; i < size; ++i){
            tensor_ptr[i] = t.tensor_ptr[i];
        }
        return *this;
    }

    // Move assignment: take ownership of t's buffer and reset t to empty.
    template<typename T>
    tensor<T>& tensor<T>::operator=(tensor &&t){
        if(this == &t){
            return *this;
        }

        if(tensor_ptr){
            delete[] tensor_ptr;
        }

        dim = t.dim;
        shape = std::move(t.shape);
        size = t.size;
        tensor_ptr = t.tensor_ptr;

        t.dim = 0;
        t.size = 0;
        t.tensor_ptr = nullptr;

        return *this;
    }

    // General tensor element access.  Bounds are checked in debug builds.
    template<typename T>
    T& tensor<T>::operator()(const std::vector<int> &index){
        return tensor_ptr[offset(index)];
    }

    // Const general tensor element access.
    template<typename T>
    const T& tensor<T>::operator()(const std::vector<int> &index) const{
        return tensor_ptr[offset(index)];
    }

    // Three-dimensional access is the common case for RI coefficients.
    template<typename T>
    T& tensor<T>::operator()(const int i, const int j, const int k){
        assert(dim == 3);
        assert(i >= 0 && i < shape[0]);
        assert(j >= 0 && j < shape[1]);
        assert(k >= 0 && k < shape[2]);
        return tensor_ptr[(i * shape[1] + j) * shape[2] + k];
    }

    // Const three-dimensional access.
    template<typename T>
    const T& tensor<T>::operator()(const int i, const int j, const int k) const{
        assert(dim == 3);
        assert(i >= 0 && i < shape[0]);
        assert(j >= 0 && j < shape[1]);
        assert(k >= 0 && k < shape[2]);
        return tensor_ptr[(i * shape[1] + j) * shape[2] + k];
    }

    // In-place scalar multiplication.
    template<typename T>
    void tensor<T>::operator*=(const T &s){
        for(int i = 0; i < size; ++i){
            tensor_ptr[i] *= s;
        }
    }

    // In-place scalar addition.
    template<typename T>
    void tensor<T>::operator+=(const T &s){
        for(int i = 0; i < size; ++i){
            tensor_ptr[i] += s;
        }
    }

    // In-place tensor addition.  Shapes must match element by element.
    template<typename T>
    void tensor<T>::operator+=(const tensor &t){
        assert(dim == t.dim);
        assert(shape == t.shape);
        for(int i = 0; i < size; ++i){
            tensor_ptr[i] += t.tensor_ptr[i];
        }
    }

    // In-place scalar subtraction.
    template<typename T>
    void tensor<T>::operator-=(const T &s){
        for(int i = 0; i < size; ++i){
            tensor_ptr[i] -= s;
        }
    }

    // In-place tensor subtraction.  Shapes must match element by element.
    template<typename T>
    void tensor<T>::operator-=(const tensor &t){
        assert(dim == t.dim);
        assert(shape == t.shape);
        for(int i = 0; i < size; ++i){
            tensor_ptr[i] -= t.tensor_ptr[i];
        }
    }

    // Tensor equality: first compare shape, then compare every stored element.
    template<typename T>
    bool tensor<T>::operator==(const tensor &t){
        if(dim != t.dim || shape != t.shape || size != t.size){
            return false;
        }
        for(int i = 0; i < size; ++i){
            if(!(tensor_ptr[i] == t.tensor_ptr[i])){
                return false;
            }
        }
        return true;
    }

    // Tensor + tensor.  A new tensor is returned; inputs are not changed.
    template<typename T>
    tensor<T> operator+(const tensor<T> &t1, const tensor<T> &t2){
        assert(t1.dim == t2.dim);
        assert(t1.shape == t2.shape);

        tensor<T> result(t1.shape);
        for(int i = 0; i < result.size; ++i){
            result.tensor_ptr[i] = t1.tensor_ptr[i] + t2.tensor_ptr[i];
        }
        return result;
    }

    // Tensor - tensor.  A new tensor is returned; inputs are not changed.
    template<typename T>
    tensor<T> operator-(const tensor<T> &t1, const tensor<T> &t2){
        assert(t1.dim == t2.dim);
        assert(t1.shape == t2.shape);

        tensor<T> result(t1.shape);
        for(int i = 0; i < result.size; ++i){
            result.tensor_ptr[i] = t1.tensor_ptr[i] - t2.tensor_ptr[i];
        }
        return result;
    }

    // Tensor * tensor is element-wise multiplication.
    // A generic tensor has no single matrix-like product, so keep this simple.
    template<typename T>
    tensor<T> operator*(const tensor<T> &t1, const tensor<T> &t2){
        assert(t1.dim == t2.dim);
        assert(t1.shape == t2.shape);

        tensor<T> result(t1.shape);
        for(int i = 0; i < result.size; ++i){
            result.tensor_ptr[i] = t1.tensor_ptr[i] * t2.tensor_ptr[i];
        }
        return result;
    }

    // Scalar * tensor.
    template<typename T>
    tensor<T> operator*(const T &s, const tensor<T> &t){
        tensor<T> result(t.shape);
        for(int i = 0; i < result.size; ++i){
            result.tensor_ptr[i] = s * t.tensor_ptr[i];
        }
        return result;
    }

    // Tensor * scalar.
    template<typename T>
    tensor<T> operator*(const tensor<T> &t, const T &s){
        tensor<T> result(t.shape);
        for(int i = 0; i < result.size; ++i){
            result.tensor_ptr[i] = t.tensor_ptr[i] * s;
        }
        return result;
    }

    // Print the flat tensor storage with its shape.  This keeps the function
    // generic and avoids special cases for 2D, 3D, and higher dimensions.
    template<typename T>
    void print_tensor(LibBSE::MpiComm MPI_COMM, const char *name, const tensor<T> &ten){
        std::ostringstream os;

        os << name << " shape = [";
        for(int i = 0; i < ten.dim; ++i){
            if(i){
                os << ", ";
            }
            os << ten.shape[static_cast<std::size_t>(i)];
        }
        os << "] values = [\n  ";
        for(int i = 0; i < ten.size; ++i){
            os << std::setw(16) << ten.tensor_ptr[i];
            if(i + 1 < ten.size){
                os << " ";
            }
        }
        os << "\n]\n";

        LibBSE_printf_root(MPI_COMM, "%s", os.str().c_str());
    }


}
