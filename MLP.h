//MLP c++
#ifndef MLP_H
#define MLP_H

#include <math.h>
#include <vector>
#include <fstream>
#include <assert.h>
#include <stdlib.h>

using namespace std;
//Simple MultiLayerPerceptron, only forward prop (no backward or fitting!)
//Purpose it to load weights from (python trained) sklearn 
//Then just forward propagate to use in ADST files for example
//Right now it only support n hidden layers that have the same size
//About 3-4 times faster than predicting from python 
class MLP {
	public:
		MLP(int number_of_layers, int layer_sizes, 
				int input_dim, int output_dim);	
		~MLP();

		vector<double> Predict(vector<double> X);
		void LoadWeights(const char* weights_file);
		vector<double> GetBiass();
		vector<double> GetWeights();

	private:
		unsigned int number_of_layers_;
		unsigned int layer_sizes_;
		unsigned int input_dim_;
		unsigned int output_dim_;
		vector<double> biass_;
		vector<double> weights_;
		vector<vector<double>> nodes_;
		vector<double> input_layer_;

};

#endif
