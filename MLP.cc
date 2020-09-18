#include <MLP.h>

MLP::MLP(int number_of_layers, int layer_sizes, 
		int input_dim, int output_dim) {
	number_of_layers_ = number_of_layers;
	layer_sizes_ = layer_sizes;
	input_dim_ = input_dim;
	output_dim_ = output_dim;
	input_layer_.reserve(input_dim_);
	vector<float> output_layer_(output_dim_); //This needs to be set like this to init size...
	
	//Nodes holds all the forward propagated node values
	//1 vector for every layer (including input and output)
	nodes_.push_back(input_layer_);
	for (int i =0; i < number_of_layers_; i++) {
		vector<float> hidden_layer(layer_sizes_);
		nodes_.push_back(hidden_layer);
	}

	nodes_.push_back(output_layer_);
}

void MLP::LoadWeights(const char* bias_filename, const char* weights_filename) {

	//Load weights from two txt files
	//Both txt files, just contain the weights in a standard ordered 1 column

	int nbias = number_of_layers_*layer_sizes_ + output_dim_;
	int nweights = input_dim_*layer_sizes_ + 
		(number_of_layers_-1)*layer_sizes_*layer_sizes_ + 
		layer_sizes_*output_dim_;


	ifstream bias_file(bias_filename);
	ifstream weights_file(weights_filename);

	float tmp;

	for (int i = 0; i < nbias; i++) {
		bias_file >> tmp;
		biass_.push_back(tmp);
	}

	for (int i = 0; i < nweights; i++) {
		weights_file >> tmp;
		weights_.push_back(tmp);
	}

	
}


vector<float> MLP::Predict(vector<float> X) {
	//Predict trace (or signal) based on input X 
	//X contains typically, lgE, cos_theta, Xmax, r, cos_psi
	//
    assert (X.size() == input_dim_);

	nodes_[0] = X;
	//Loop over layers /
	float output;
	int n = 0;	
	int m = 0;
	//Matrix multiplications for every layer: y = tanh(w*X+b)
	int nloops = nodes_.size() - 1;
	vector<float> this_layer;
	vector<float> next_layer;
	int this_layer_size;
	int next_layer_size;
	for (int i = 0; i <  nloops; i++){
		this_layer = nodes_[i];
		next_layer = nodes_[i+1];
		this_layer_size = this_layer.size();
		next_layer_size = next_layer.size();
		
		//Loop over nodes, this is just matrix multiplication
		for (int j = 0; j < next_layer_size; j++){
			output = biass_[n];
			n++;
			//Loop over previous (input) layer
			for (int k = 0; k < this_layer_size; k++) {
				output += weights_[m] * nodes_[i][k];
				m++;
			}	
			//Output layer no activation
			if (i < number_of_layers_) {
				nodes_[i+1][j] = tanh(output);
			} else {
				nodes_[i+1][j] = output;
			}
		}
	}


	return nodes_.back(); //return latest node, is output layer

}

MLP::~MLP(){};


