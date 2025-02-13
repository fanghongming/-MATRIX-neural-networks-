#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "Parameter.h"
template <typename Scalar,int Rows,int Cols>
struct  Optimizer
{
    Parameter<Scalar,Rows,Cols>&param;
    Optimizer(Parameter<Scalar,Rows,Cols>&p):param(p)
    {

    }
    virtual void step()=0;
};
#endif 