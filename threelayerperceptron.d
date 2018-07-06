import std.stdio;
import std.math;
import std.random;
import std.algorithm;

auto sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}
auto sigmoid_d(double x)
{
  auto y = sigmoid(x);
  return y * (1.0 - y);
}

struct PredicateResult
{
  public:
    double[] hidden_input;
    double[] hidden_us;
    double[] hidden_output;
    double[] output_input;
    double[] output_us;
    double[] output_output;
}

class ThreeLayerPerceptron
{
  public:
    int input_layer_size;
    int hidden_layer_size;
    int output_layer_size;

    double[][] input_hidden_weights;
    double[][] hidden_output_weights;

    double learn_rate;

    this(int input_layer_size, int hidden_layer_size, int output_layer_size, double learn_rate)
    {
      this.learn_rate = learn_rate;
      this.input_layer_size = input_layer_size;
      this.hidden_layer_size = hidden_layer_size;
      this.output_layer_size = output_layer_size;

      this.input_hidden_weights = new double[][](hidden_layer_size, input_layer_size + 1);
      this.hidden_output_weights = new double[][](output_layer_size, hidden_layer_size + 1);

      Random gen = rndGen();
      foreach (i; 0..hidden_layer_size) {
        foreach (j; 0..input_layer_size + 1) {
          input_hidden_weights[i][j] = uniform(-1.0, 1.0, gen);
        }
      }

      foreach (i; 0..output_layer_size) {
        foreach (j; 0..hidden_layer_size + 1) {
          hidden_output_weights[i][j] = uniform(-1.0, 1.0, gen);
        }
      }
    }

    auto predicate(double[] input)
    in
    {
      assert(input.length == this.input_layer_size);
    }
    do
    {
      input = [1.0] ~ input.dup;
      auto hidden_us = new double[](hidden_layer_size);
      hidden_us.fill(0.0);
      foreach (i; 0..hidden_layer_size) {
        foreach (j; 0..input_layer_size + 1) {
          hidden_us[i] += input_hidden_weights[i][j] * input[j];
        }
      }

      auto hidden_output = new double[](hidden_layer_size);
      foreach (i, u; hidden_us) {
        hidden_output[i] = sigmoid(u);
      }

      auto output_input = [1.0] ~ hidden_output;
      auto output_us = new double[](output_layer_size);
      output_us.fill(0.0);

      foreach (i; 0..output_layer_size) {
        foreach (j; 0..hidden_layer_size + 1) {
          output_us[i] += hidden_output_weights[i][j] * output_input[j];
        }
      }

      auto output_output = new double[](output_layer_size);
      foreach (i, u; output_us) {
        output_output[i] = sigmoid(u);
      }

      return PredicateResult(input, hidden_us, hidden_output, output_input, output_us, output_output);
    }

    void training(double[] input, double[] expect)
    in
    {
      assert(input.length == input_layer_size);
      assert(expect.length == output_layer_size);
    }
    do
    {
      auto result = predicate(input);

      auto deltas = new double[](hidden_layer_size + 1);
      deltas.fill(0.0);
      auto hidden_output_d = new double[][](output_layer_size, hidden_layer_size + 1);
      foreach (i; 0..output_layer_size) {
        foreach (j; 0..hidden_layer_size + 1) {
          hidden_output_d[i][j] = (expect[i] - result.output_output[i]) * sigmoid_d(result.output_us[i]) * result.output_input[j];
          if (j != hidden_layer_size) {
            deltas[j] += (expect[i] - result.output_output[i]) * sigmoid_d(result.output_us[i]) * hidden_output_weights[i][j];
          }
        }
      }

      auto input_hidden_d = new double[][](hidden_layer_size, input_layer_size + 1);
      foreach (i; 0..hidden_layer_size) {
        foreach (j; 0..input_layer_size + 1) {
          input_hidden_d[i][j] = 0;
          input_hidden_d[i][j] = deltas[i] * sigmoid_d(result.hidden_us[i]) * result.hidden_input[j];
        }
      }

      foreach (i; 0..output_layer_size) {
        foreach (j; 0..hidden_layer_size + 1) {
          hidden_output_weights[i][j] += learn_rate * hidden_output_d[i][j];
        }
      }

      foreach (i; 0..hidden_layer_size) {
        foreach (j; 0..output_layer_size + 1) {
          input_hidden_weights[i][j] += learn_rate * input_hidden_d[i][j];
        }
      }
    }
}


void main()
{
  auto tlp = new ThreeLayerPerceptron(2, 30, 1, 0.5);

  const auto T = 1.00;
  const auto F = 0.01;

  foreach(i; 0..3000) {
    tlp.training([F, F], [F]);
    tlp.training([F, T], [T]);
    tlp.training([T, F], [T]);
    tlp.training([T, T], [F]);
  }

  writeln(tlp.predicate([F, F]).output_output);
  writeln(tlp.predicate([F, T]).output_output);
  writeln(tlp.predicate([T, F]).output_output);
  writeln(tlp.predicate([T, T]).output_output);
}
