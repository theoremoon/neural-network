import std.stdio;
import std.math;
import std.random;
import std.range;
import std.algorithm;
import std.functional;

auto sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}
auto sigmoid_d(double x)
{
  auto y = sigmoid(x);
  return y * (1.0 - y);
}
auto tanh_d(double x)
{
  return 1.0 - pow(tanh(x), 2);
}
auto relu(double x)
{
  return (x > 0) ? x : 0.0;
}
auto relu_d(double x)
{
  return (x > 0) ? 1.0 : 0.0;
}

struct ForwardResult 
{
  public:
    double[] input;
    double[] us;
    double[] output;
}

alias activateT = double delegate(double);
alias activateFT = double function(double);
class Layer
{
  public:
    uint input_layer_size;
    uint output_layer_size;

    double[][] weights;

    activateT activate;
    activateT derived;

    double learn_rate;

    this (uint input_layer_size, uint output_layer_size, activateT activate, activateT derived, double learn_rate)
    {
      this.learn_rate = learn_rate;
      this.activate = activate;
      this.derived = derived;

      this.input_layer_size = input_layer_size;
      this.output_layer_size = output_layer_size;
      this.weights = new double[][](output_layer_size, input_layer_size + 1);   // + 1 is weight for bias

      auto rnd = rndGen();
      foreach (i; 0..output_layer_size) {
        foreach (j; 0..input_layer_size + 1) {
          this.weights[i][j] = uniform(-1.0, 1.0, rnd);
        }
      }
    }

    auto forward(double[] input)
    in
    {
      assert(input.length == this.input_layer_size);
    }
    do
    {
      input = [1.0] ~ input.dup;  // add bias

      auto us = new double[](output_layer_size);
      us.fill(0.0);  // avoiding NaN

      foreach (i; 0..output_layer_size) {
        foreach (j; 0..input_layer_size + 1) {
          us[i] += weights[i][j] * input[j];
        }
      }

      auto output = new double[](output_layer_size);
      foreach (i, u; us) {
        output[i] = activate(u);
      }

      return ForwardResult(input, us, output);
    }

    auto backward(double[] input, double[] us, double[] deltas)
    {
      double[] next_deltas = new double[](input_layer_size);
      next_deltas.fill(0.0);
      auto weights_d = new double[][](output_layer_size, input_layer_size + 1);

      foreach (i; 0..output_layer_size) {
        foreach (j; 0..input_layer_size + 1) {
          weights_d[i][j] = deltas[i] * sigmoid_d(us[i]) * input[j];
          if (j != input_layer_size) {
            next_deltas[j] += deltas[i] * weights[i][j];
          }
        }
      }

      foreach (i; 0..output_layer_size) {
        foreach (j; 0..input_layer_size + 1) {
          weights[i][j] += learn_rate * weights_d[i][j];
        }
      }

      return next_deltas;
    }
}

class MultiLayerPerceptron
{
  public:
    Layer[] layers = [];
    uint input_layer_size = 0;
    uint output_layer_size = 0;

    this(uint input_layer_size)
    {
      this.input_layer_size = input_layer_size;
      this.output_layer_size = input_layer_size;
    }

    void addLayer(uint output_layer_size, activateFT activate, activateFT derived, double learn_rate)
    {
      this.addLayer(output_layer_size, activate.toDelegate, derived.toDelegate, learn_rate);
    }
    void addLayer(uint output_layer_size, activateT activate, activateT derived, double learn_rate)
    {
      layers ~= new Layer(this.output_layer_size, output_layer_size, activate, derived, learn_rate);
      this.output_layer_size = output_layer_size;
    }

    auto predicate(double[] input)
    in
    {
      assert(input.length == this.input_layer_size);
    }
    do
    {
      auto next = ForwardResult([], [], input);
      foreach (layer; this.layers) {
        next = layer.forward(next.output);
      }
      return next.output;
    }

    void training(double[][] inputs, double[][] expects)
    in
    {
      assert(inputs.length == expects.length);
    }
    do
    {
      ulong[] indexes = iota(0, inputs.length).array;
      foreach (i; indexes.randomShuffle) {
        training_one(inputs[i], expects[i]);
      }
    }

    void training_one(double[] input, double[] expect)
    in
    {
      assert(input.length == this.input_layer_size);
    }
    do
    {
      ForwardResult[] forwarded = [ForwardResult([], [], input)];
      foreach (layer; this.layers) {
        forwarded ~= layer.forward(forwarded[$-1].output);
      }

      double[] deltas = [];
      foreach (i; 0..output_layer_size) {
        deltas ~= expect[i] - forwarded[$-1].output[i];
      }

      for (long i = this.layers.length - 1; i >= 0; i--) {
        deltas = layers[i].backward(forwarded[i+1].input, forwarded[i+1].us, deltas);
      }
    }
}


void main()
{
  auto mlp = new MultiLayerPerceptron(2);
  mlp.addLayer(10, &relu, &relu_d, 0.6);
  mlp.addLayer(1, &sigmoid, &sigmoid_d, 0.3);

  const auto T = 1.00;
  const auto F = 0.01;

  foreach(i; 0..3000) {
    mlp.training([
      [F, F],
      [F, T],
      [T, F],
      [T, T],
    ], [
      [F], [T], [T], [F]
    ]);
  }

  writeln(mlp.predicate([F, F]));
  writeln(mlp.predicate([F, T]));
  writeln(mlp.predicate([T, F]));
  writeln(mlp.predicate([T, T]));
}
