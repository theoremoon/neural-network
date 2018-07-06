import random
import math

class Layer():
    """
    ニューラルネットワークの層
    """
    def __init__(self, input_size, output_size, activation, derived, learn_rate):
        self.input_size = input_size + 1
        self.output_size = output_size
        self.learn_rate = learn_rate

        self.weights = [[2 * random.random() - 1 for _ in range(input_size + 1)] for _ in range(output_size)]
        self.activation = activation
        self.derived = derived

    def forward(self, input):
        self.input = [1] + input # 逆伝播のときに使うのでとっておく 
        self.output = []  # はい
        self.unit_inputs = []  # 逆伝播のときに使う

        for j in range(self.output_size): 
            u = 0
            for i in range(self.input_size):
                u += self.input[i] * self.weights[j][i]
            self.unit_inputs.append(u)
            self.output.append(self.activation(u))
        return self.output

    def backward(self, xs):
        deltas = []
        d_ws = []

        # W の変量を求める
        for j in range(self.output_size):
            delta_j = xs[j] * self.derived(self.unit_inputs[j])
            d_ws2 = []
            for i in range(self.input_size):
                delta_w = delta_j * self.input[i]
                d_ws2.append(delta_w)
            deltas.append(delta_j)
            d_ws.append(d_ws2)

        # 上の階層へ送るための値（xs)を求める
        r_xs = []
        for i in range(self.input_size):
            r_x = 0
            for j in range(self.output_size):
                r_x += deltas[j] * self.weights[j][i]
            r_xs.append(r_x)


        # W を更新
        for j in range(self.output_size):
            for i in range(self.input_size):
                # 勾配を下るので減算
                self.weights[j][i] -= d_ws[j][i] * self.learn_rate

        return r_xs


class NN():
    """
    ニューラルネットワークの本体
    """

    def __init__(self, input_size):
        self.layers = []
        self.lastlayer_output_size = input_size

    def addLayer(self, size, activation, derived, learn_rate):
        """
        中間層、あるいは出力層を追加する。
        出力層の場合はderivedはNoneでいい
        
        size: その layer が持つ unit の数
        activation: 活性化関数 double -> double
        derived: 活性化関数の微分 double -> double
        learn_rate: 学習率
        """
        self.layers.append(Layer(
            input_size=self.lastlayer_output_size,
            output_size=size,
            activation=activation,
            derived=derived,
            learn_rate=learn_rate))
        self.lastlayer_output_size = size

    def forward(self, input):
        """
        順伝播
        """
        last_output = input
        for layer in self.layers:
            last_output = layer.forward(last_output)
        return last_output

    def fit(self, inputs, expects):
        """
        逆伝播
        """

        outputs = self.forward(inputs)
        xs = []
        for i in range(len(outputs)):
            xs.append(outputs[i] - expects[i])

        for layer in reversed(self.layers):
            xs = layer.backward(xs)

        return outputs


def logistic(x):
    return 1 / (1 + math.exp(-x))

def logistic_d(x):
    return (1 - logistic(x)) * logistic(x)


def main():
    network = NN(input_size=2)
    network.addLayer(size=10, activation=logistic, derived=logistic_d, learn_rate=0.75)
    network.addLayer(size=1, activation=logistic, derived=logistic_d, learn_rate=0.5)

    T = 0.999
    F = 0.001
    for i in range(20000):
        r1 = network.fit(inputs=[F, F], expects=[F])
        r2 = network.fit(inputs=[F, T], expects=[T])
        r3 = network.fit(inputs=[T, F], expects=[T])
        r4 = network.fit(inputs=[T, T], expects=[F])

    print(network.forward(input=[F, F]))
    print(network.forward(input=[F, T]))
    print(network.forward(input=[T, F]))
    print(network.forward(input=[T, T]))


if __name__ == '__main__':
    main()
