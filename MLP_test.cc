#include <MLP.h>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {

  MLP mlp = MLP(2, 2, 1, 1);
  mlp.LoadWeights("test_weights.txt");

  vector<double> weights = mlp.GetWeights();
  vector<double> biass = mlp.GetBiass();

  cout << "n weights : " << weights.size() << endl;

  for (auto w : weights) {
    cout << w << endl;
  };
  cout << "n bias : " << biass.size() << endl;

  for (auto b : biass) {
    cout << b << endl;
  };

  cout << "test Predict" << endl;
  vector<double> testX = {1};

  vector<double> out = mlp.Predict(testX);

  for (auto x : out) {
    cout << x << endl;
  };

  cout << endl;

  for (auto x : out) {
    cout << x << endl;
  };

  return 0;
}
