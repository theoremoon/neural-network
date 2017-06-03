import numpy
from scipy import special

class NeuralNet(object):
    """ニューラルネットワーク"""

    def __init__(self, layer_sizes):
        """
        :param layer_sizes: 各層の大きさが詰まったリスト。最初と最後は入力層と出力層
        """

        assert len(layer_sizes) >= 2 # 少なくとも入力出力層の分はあるはず

        # 各層の間の重み行列のリスト
        self.weights = []
        for i in range(0, len(layer_sizes)-1):
            h = layer_sizes[i+1]
            w = layer_sizes[i]
            self.weights.append(numpy.random.rand(h, w)-0.5)
        
    def try_value(self, input_vector):
        """ある入力について、出力を計算する
        
        :param input_vector: 入力値。リスト
        :return: 出力値
        """

        # 入力を行列に変換する
        vs = numpy.array(input_vector, ndmin=2).T
        outputs = [vs]
        for w in self.weights:
            # 重み付けして活性化関数に突っ込む
            neruron_input = numpy.dot(w, vs)
            vs = special.expit(neruron_input)

            # 出力値を保存しておく
            outputs.append(vs)

        assert len(outputs) == len(self.weights)+1        
        self.outputs = outputs

        return numpy.squeeze(numpy.asarray(vs.T))


    def study(self, input_vector, desired_vector):
        """データから学習する
        
        :param input_vector: 入力値
        :param desired_vector: 理想値
        """

        # 学習率
        rate = 0.6

        desired_vector = numpy.array(desired_vector, ndmin=2).T

        # 計算させてみて誤差から学習する
        tried = self.try_value(input_vector)
        i = len(self.outputs)-1
        error = None
        while i > 0:
            if i == len(self.outputs)-1:
                error = desired_vector-self.outputs[i]
            else:
                error = numpy.dot(self.weights[i].T, error)
            delta = rate * numpy.dot(error * self.outputs[i] * (1 - self.outputs[i]), self.outputs[i-1].T)
            self.weights[i-1] += delta
            i-=1


    def judge(self):
        """あるデータを判定する"""
        pass

    def __repr__(self):
        s = "NeuralNet: {} layers".format(len(self.weights)+1)
        return s

def main():
    """xor の学習をしてみる（できてない"""

    # 2入力2出力
    net = NeuralNet([2,5,2])
    
    T = 1
    F = 1e-3 # 入力に0を持ってくるのはだめということ

    # 学習させる（おんなじデータで何回も学習させるのは直感的にダメそうって思う
    for i in range(100):
        net.study([F,F], [F,T])
        net.study([F,T], [T,F])
        net.study([T,F], [T,F])
        net.study([T,T], [F,T])
    
    result = net.try_value([F,F])
    print("result: {} <-- {}".format(result, result[0]<result[1]))
    result = net.try_value([T,F])
    print("result: {} <-- {}".format(result, result[0]>result[1]))
 
main()