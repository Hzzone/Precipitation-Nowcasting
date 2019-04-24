/*!
*  Copyright (c) 2017 by Contributors
* \file c_api.h
* \brief C API of VarFlow
*/
#ifndef VARFLOW_C_API_H_
#define VARFLOW_C_API_H_


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#ifdef _WIN32
#define VARFLOW_DLL __declspec(dllexport)
#else
#define VARFLOW_DLL
#endif

VARFLOW_DLL void varflow(int width, int height, int max_level, int start_level, int n1, int n2,
  float rho, float alpha, float sigma, void* U, void* V,
  void* I1, void* I2);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif