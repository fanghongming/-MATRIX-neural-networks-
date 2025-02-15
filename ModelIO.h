#ifndef MODEL_IO_H
#define MODEL_IO_H
#include <fstream>
#include "NeuralNetwork.h"
template<typename Scalar>
void save_model( NeuralNetwork<Scalar>&network,const std::string&filename)
{
    std::fstream file(filename,std::ios::in|std::ios::out|std::ios::trunc);
    //std::cout<<"ok"<<"\n";
    for( auto &module:network.get_modules())
    {
        module->save(file);
    }
    file.close();
}
template<typename Scalar>
void load_model(NeuralNetwork<Scalar>&network,std::string filename)
{
    std::fstream file(filename,std::ios::in|std::ios::out);//这里面不能用trunc

    for(auto &module:network.get_modules())
    {
        module->load(file);
    }
    file.close();
}
#endif // MODEL_IO_H