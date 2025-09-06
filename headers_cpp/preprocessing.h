#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <thread>

#include "matrix.h"
#include "read_file.h"
#include "streamlines.h"


Matrix orthonormalise( const Matrix& mat, Matrix& X, Matrix& Y, Matrix& Z, const Shape* new_shape=nullptr );



#endif
