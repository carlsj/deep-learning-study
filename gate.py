import numpy as np


class GATE:
    def __init__(self):
        self.w = {'and': np.array([0.5, 0.5]), 'or': np.array([0.5, 0.5]), 'nand': np.array([-0.5, -0.5])}
        self.b = {'and': -0.7, 'or': -0.2, 'nand': 0.7}

    @staticmethod
    def check_input(x1, x2):
        if type(x1) != type(x2):
            print("Input type is not matched")
            return False

        input_range = [0, 1]
        if (x1 in input_range) and (x2 in input_range):
            return True
        else:
            print('Input is not in a range of' + str(input_range))
            return False

    @staticmethod
    def gate_operation(x, w, b):
        tmp = np.sum(x * w) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def and_op(self, x1, x2):
        if self.check_input(x1, x2):
            x = np.array([x1, x2])
            w = self.w['and']
            b = self.b['and']
            return self.gate_operation(x, w, b)
        else:
            return False

    def or_op(self, x1, x2):
        if self.check_input(x1, x2):
            x = np.array([x1, x2])
            w = self.w['or']
            b = self.b['or']
            return self.gate_operation(x, w, b)
        else:
            return False

    def nand_op(self, x1, x2):
        if self.check_input(x1, x2):
            x = np.array([x1, x2])
            w = self.w['nand']
            b = self.b['nand']
            return self.gate_operation(x, w, b)
        else:
            return False

    def xor_op(self, x1, x2):
        if self.check_input(x1, x2):
            s1 = self.nand_op(x1, x2)
            s2 = self.or_op(x1, x2)
            return self.and_op(s1, s2)
        else:
            return False

if __name__ == '__main__':
    gate_sample = GATE()
    in_sample = [(0, 0), (1, 0), (0, 1), (1, 1)]

    print('AND operations')
    for xs in in_sample:
        y = gate_sample.and_op(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
    print()

    print('OR operations')
    for xs in in_sample:
        y = gate_sample.or_op(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
    print()

    print('NAND operations')
    for xs in in_sample:
        y = gate_sample.nand_op(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
    print()

    print('XOR operations')
    for xs in in_sample:
        y = gate_sample.xor_op(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
    print()

    print('Test input conditions')
    y = gate_sample.and_op(0.1, 0)
    print(y)
    y = gate_sample.or_op('t', 1)
    print(y)
    y = gate_sample.nand_op(True, 0)
    print(y)
    y = gate_sample.xor_op(0, 7)
    print(y)



