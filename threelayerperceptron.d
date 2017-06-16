import std.random;
import std.math;
import std.stdio;
import std.algorithm;


class ThreeLayerPerceptron {
    public {
        int InputLayerSize, HiddenLayerSize, OutputLayerSize;
        float[][] Input_Hidden_Weight, Hidden_Output_Weight;
        float[] InputOutput, HiddenOutput, OutOutput;
        float LearningRate;
    }
    this(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, float learningRate) {
        InputLayerSize = inputLayerSize;
        HiddenLayerSize = hiddenLayerSize;
        OutputLayerSize = outputLayerSize;
        LearningRate = learningRate;

        Input_Hidden_Weight = new float[][](HiddenLayerSize, InputLayerSize);
        Hidden_Output_Weight = new float[][](OutputLayerSize, HiddenLayerSize);

        Random gen = rndGen();
        for (int i = 0; i < HiddenLayerSize; i++) {
            for (int j = 0; j < InputLayerSize; j++) {
                Input_Hidden_Weight[i][j] = uniform(0f, 1f, gen) - 0.5;
            }
        }
        for (int i = 0; i < OutputLayerSize; i++) {
            for (int j = 0 ; j < HiddenLayerSize; j++) {
                Hidden_Output_Weight[i][j] = uniform(0f, 1f, gen) - 0.5;
            }
        }
    }
    double Sigmoid(double x) {
        return 1f / (1+exp(-x));
    }

    float[] Classfy(float[] input) {
        if (input.length != InputLayerSize) {
            return [];
        }
        InputOutput = input.dup;

        float[] hiddenInput = new float[](HiddenLayerSize);
        hiddenInput.fill(0f);
        for (int i = 0; i < HiddenLayerSize; i++) {
            for (int j = 0; j < InputLayerSize; j++) {
                hiddenInput[i] += InputOutput[j] * Input_Hidden_Weight[i][j];
            }
        }

        HiddenOutput = new float[](HiddenLayerSize);
        foreach (i, v; hiddenInput) {
            HiddenOutput[i] = Sigmoid(v);
        }

        float[] outInput = new float[](OutputLayerSize);
        outInput.fill(0f);
        for (int i = 0; i < OutputLayerSize; i++) {
            for (int j = 0; j < HiddenLayerSize; j++) {
                outInput[i] += HiddenOutput[j] * Hidden_Output_Weight[i][j];
            }
        }

        OutOutput = new float[](OutputLayerSize);
        foreach (i, v; outInput) {
            OutOutput[i] = Sigmoid(v);
        }

        return OutOutput;
    }

    void Training(float[] input, float[] desired) {
        if (input.length != InputLayerSize || desired.length != OutputLayerSize) {
            return;
        }

        Classfy(input);

        float[][] hidden_Output_Weight_d = new float[][](OutputLayerSize, HiddenLayerSize);
        for (int i = 0; i < OutputLayerSize; i++) {
            auto memo = (OutOutput[i]-desired[i]) * OutOutput[i] * (1-OutOutput[i]);
            for (int j = 0; j < HiddenLayerSize; j++) {
                hidden_Output_Weight_d[i][j] =  memo * HiddenOutput[j];
            }
        }

        float[][] input_Hidden_Weight_d = new float[][](HiddenLayerSize, InputLayerSize);
        for (int i = 0; i < HiddenLayerSize; i++) {
            for (int j = 0; j < InputLayerSize; j++) {
                input_Hidden_Weight_d[i][j] = 0f;
                for (int k = 0; k < OutputLayerSize; k++) {
                    input_Hidden_Weight_d[i][j] += (OutOutput[k]-desired[k]) * OutOutput[k] * (1-OutOutput[k]) * Hidden_Output_Weight[k][i];
                }
                input_Hidden_Weight_d[i][j] *= HiddenOutput[i] * (1-HiddenOutput[i]) * InputOutput[j];
            }
        }

        foreach(i,v; hidden_Output_Weight_d) {
            foreach (j,w; v) {
                Hidden_Output_Weight[i][j] += -this.LearningRate * w;
            }
        }
        foreach(i,v; input_Hidden_Weight_d) {
            foreach (j,w;v) {
                Input_Hidden_Weight[i][j] += -this.LearningRate * w;
            }
        }
    }
}

void main()
{
    ThreeLayerPerceptron mlp = new ThreeLayerPerceptron(2, 10, 1, 0.3);

    foreach(i; 0..3000) {
        mlp.Training([1f, 1f], [0.01f]);
        mlp.Training([1f, 0.01f], [1f]);
        mlp.Training([0.01f, 1f], [1f]);
        mlp.Training([0.01f, 0.01f], [0.01f]);
    }

    writeln("==RESULT==");
    writeln(mlp.Classfy([1f,1f]));
    writeln(mlp.Classfy([1f,0.01f]));
    writeln(mlp.Classfy([0.01f,1f]));
    writeln(mlp.Classfy([0.01f,0.01f]));

}