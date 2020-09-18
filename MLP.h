//MLP c++

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

		vector<float> Predict(vector<float> X);
		void LoadWeights(const char* bias_file, const char* weights_file);

	private:
		unsigned int number_of_layers_;
		unsigned int layer_sizes_;
		unsigned int input_dim_;
		unsigned int output_dim_;
		vector<float> biass_;
		vector<float> weights_;
		vector<vector<float>> nodes_;
		vector<float> input_layer_;

};
