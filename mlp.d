import std.random;
import std.math;
import std.stdio;
import std.algorithm;


class MLP {
    public {
        int[] LayerSizes;
        float[][][] Weights;
        float[][] Outputs;
        float LearningRate;
    }
    this(int[] layerSizes, float learningRate) {
        if (layerSizes.length < 2) {
            throw new Exception("Two or more layers required");
        }
        LayerSizes = layerSizes.dup;
        LearningRate = learningRate;

        Weights.length = LayerSizes.length-1;
        Random gen = rndGen();
        for (int i = 0; i < Weights.length; i++) {
            Weights[i].length = LayerSizes[i+1];
            for (int j = 0; j < LayerSizes[i+1]; j++) {
                Weights[i][j].length = LayerSizes[i];
                for (int k = 0; k < LayerSizes[i]; k++) {
                    Weights[i][j][k] = uniform(0f,1f,gen)-0.5;
                }
            }
        }
    }
    double Sigmoid(double x) {
        return 1f / (1+exp(-x));
    }

    float[] Classfy(float[] input) {
        if (input.length != LayerSizes[0]) {
            throw new Exception("Input layer size mismatched");
        }
        Outputs.length = LayerSizes.length;
        Outputs[0] = input.dup;

        for (int i = 1; i < Outputs.length; i++) {
            Outputs[i].length = LayerSizes[i];
            for (int j = 0; j < LayerSizes[i]; j++) {
                float u = 0f;
                for (int k = 0; k < LayerSizes[i-1]; k++) {
                    u += Outputs[i-1][k] * Weights[i-1][j][k];
                }
                Outputs[i][j] = Sigmoid(u);
            }
        }

        return Outputs[$-1];
    }

    void Training(float[] input, float[] desired) {
        if (input.length != LayerSizes[0] || desired.length != LayerSizes[$-1]) {
            throw new Exception("Input or output layer size mismatched");
        }

        Classfy(input);

        float[][][] Weight_ds;
        Weight_ds.length = LayerSizes.length-1;

        // すべての重みを更新する
        // Weight_ds[i]: i層からi+1層への重み
        // j: 重みの行数（i+1層の大きさ）
        // k: 重みの列数（i層の大きさ）
        // l: i+1層からi+2層への重みの行数（i+2層の大きさ） 
        for (int i = Weight_ds.length-1; i >= 0; i--) {
            // 出力層
            if (i == Weight_ds.length-1) {
                Weight_ds[i].length = LayerSizes[i+1];
                for (int j = 0; j < Weight_ds[i].length; j++) {
                    auto memo = (Outputs[i+1][j] - desired[j]) * Outputs[i+1][j] * (1-Outputs[i+1][j]);
                    Weight_ds[i][j].length = LayerSizes[i];
                    for (int k = 0; k < Weight_ds[i][j].length; k++) {
                        Weight_ds[i][j][k] = memo * Outputs[i][k];
                    }
                }
            }

            // それ以外
            else {
                Weight_ds[i].length = LayerSizes[i+1];
                for (int j = 0; j < Weight_ds[i].length; j++) {
                    Weight_ds[i][j].length = LayerSizes[i];
                    for (int k = 0; k < Weight_ds[i][j].length; k++) {
                        Weight_ds[i][j][k] = 0f;
                        for (int l = 0; l < LayerSizes[i+2]; l++) {
                            Weight_ds[i][j][k] += Weight_ds[i+1][l][j] * Weights[i+1][l][j];
                        }
                        Weight_ds[i][j][k] *= (1-Outputs[i+1][j]) * Outputs[i][k];
                    }
                }
            }
        }
        foreach(i,u; Weight_ds) {
            foreach(j,v; u) {
                foreach (k,w;v) {
                    Weights[i][j][k] += -this.LearningRate * w;
                }
            }
        }
    }
}

void main()
{
    MLP mlp = new MLP([2, 10, 10, 1], 0.3);

    foreach(i; 0..10000) {
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
