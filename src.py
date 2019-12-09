from itertools import zip_longest
import scipy.special
import itertools
import pickle
import random
import os

DEBUG = True

class LinearCode:

    def __init__(self, n, k, d, channel_error_prob=0.01):
        if n <= k or k <= 0 or d <= 0 or ((d - 1) // 2 > n):
            raise ValueError('Invalid LinearCode parameters')
        if not self._satisfies_gilbert_varshamov_bound(n, k, d):
            raise ValueError('Gilbert-Varshamov bound conditions is not satisfied')

        self._n = n
        self._k = k
        self._r = self._n - self._k
        self._d = d
        self._p = channel_error_prob
        self.decode_error_prob = self._error_prob(self._p, self._n, self._d)

    def _satisfies_gilbert_varshamov_bound(self, n, k, d):
        s = 0
        for i in range(d - 1):
            s += self._comb(n - 1, i)
        return s < (1 << (n - k))

    def _error_prob(self, p, n, d):
        t = (d - 1) // 2
        error_prob = 0
        for i in range(t + 1):
            error_prob += self._comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        return 1 - error_prob

    def _comb(self, N, k):
        return scipy.special.comb(N, k, exact=True)

    def _get_combinations(self, d, r):
        elements = []
        for i in range(1, d - 1):
            elements += list(itertools.combinations([1 << (r - 1 - j) for j in range(r)], i))
        return elements

    def _reduce_xor(self, l):
        res = 0
        for elem in l:
            res ^= elem
        return res

    def _transpose(self, matrix, size):
        matrix_t = [0] * size
        n = len(matrix) - 1
        for i, column in enumerate(matrix):
            for row in range(size):
                mask = 1 << row
                matrix_t[size - 1 - row] += int((column & mask) == mask) << (n - i)
        return matrix_t

    def _hamming_weight(self, decimal):
        res = 0
        while decimal:
            res += decimal & 1
            decimal = decimal >> 1
        return res

    def _get_syndrome(self, y):
        zip_vec = (zip_longest(self._Ht, [y], fillvalue=y))
        return self._to_decimal(list(map(self._binary_mult, list(zip_vec))))

    def _binary_mult(self, args):
        vec_a, vec_b = args
        res_vec = vec_a & vec_b
        return self._hamming_weight(res_vec) % 2

    def _to_decimal(self, binary):
        return sum(list(map(lambda x: x[1] << (len(binary) - 1 - x[0]), [(i, b) for i, b in enumerate(binary)])))

    def _random_error(self, n, t, length):
        errors = []
        for i in range(length):
            rand_error = [1 for _ in range(random.randint(0, t))]
            rand_error += [0] * (n - len(rand_error))
            random.shuffle(rand_error)
            errors.append(to_decimal(rand_error))
        return errors

    def _get_codeword(self, message):
        zip_vec = (zip_longest(self._A, [message], fillvalue=message))
        return message << self._r | self._to_decimal(list(map(self._binary_mult, list(zip_vec))))

    def generate_A(self):
        A_t = []
        linear_comb = self._get_combinations(self._d, self._r)
        linear_dependent_vectors = set(map(self._reduce_xor, linear_comb))

        for i in range(self._k):
            new_vec = random.randint(3, (1 << self._r) - 1)
            while new_vec in linear_dependent_vectors:
                new_vec = random.randint(3, (1 << self._r) - 1)
            A_t.append(new_vec)

            if (self._d - 1 > 2):
                self._update_linear_dependent_vectors(new_vec, linear_dependent_vectors )
            linear_dependent_vectors.add(new_vec)

        self._A = self._transpose(A_t, self._r)

    def generate_Ht(self):
        self.generate_A()
        self._Ht = []
        for i, column in enumerate(self._A):
            self._Ht.append((column << self._r) | (1 << self._r - 1 - i))

    def generate_S(self):
        vectors_weight = dict()
        for i in range(1 << self._n):
            w_i = self._hamming_weight(i)
            if w_i <= (self._d - 1) // 2:
                vectors_weight.setdefault(w_i, []).append(i)
        vectors_leaders = [v for val in vectors_weight.values() for v in val]

        self._S = dict()
        self._S[0] = [0]

        for leader in vectors_leaders:
            syndrome = self._get_syndrome(leader)
            if syndrome > 0:
                self._S.setdefault(syndrome, []).append(leader)

    def generate_code(self):
        self.generate_Ht()
        self.generate_S()

    def encode(self, message, error):
        message_to_encode = [int(message[i: i + self._k], 2) for i in range(0, len(message), self._k)]
        if error is None:
            errors = self._random_error(self._n, (self._d - 1) // 2, len(message_to_encode))
        else:
            if len(error) > self._n:
                raise ValueError('Invalid error length')
            errors = [int(error, 2)] * len(message_to_encode)

        return [self._get_codeword(m) ^ e for m, e in zip(message_to_encode, errors)], errors


    def decode(self, message):
        message_to_decode = [int(message[i: i + self._n], 2) for i in range(0, len(message), self._n)]
        errors = [self._S.get(self._get_syndrome(m), [])[0] for m in message_to_decode]
        return [(m ^ e) >> self._r for m, e in zip(message_to_decode, errors)], errors






def generate(args):
    if DEBUG:
        print("Activated generator mode with parameters:")
        print(args)

    linear_code = LinearCode(n=args.n, k=(args.n - args.r), d=(2 * args.t + 1))
    linear_code.generate_code()

    out_path = os.path.normpath(os.path.expanduser(args.out))
    with open(out_path, 'wb') as handle:
        pickle.dump(linear_code, handle, protocol=pickle.HIGHEST_PROTOCOL)


def encode(args):
    if DEBUG:
        print("Activated encode mode with parameters:")
        print(args)
    file_path = os.path.normpath(os.path.expanduser(args.inputfile))
    with open(file_path, 'rb') as handle:
        linear_code = pickle.load(handle)

    encoded_message, error = linear_code.encode(args.m, args.e)
    print("Encoded message with errors:         \t{}".format('|'.join(['{0:>0{width}b}'.format(m, width=linear_code._n) for m in encoded_message])))
    print("Error:                          \t{}".format('|'.join(['{0:>0{width}b}'.format(e, width=linear_code._n) for e in error])))

def decode(args):
    if DEBUG:
        print("Activated decode mode with parameters:")
        print(args)
    file_path = os.path.normpath(os.path.expanduser(args.inputfile))
    with open(file_path, 'rb') as handle:
        linear_code = pickle.load(handle)

    decoded_message, error = linear_code.decode(args.y)
    print("Decoded message with errors:          \t{}".format('|'.join(['{0:>0{width}b}'.format(m, width=linear_code._k) for m in decoded_message])))
    print("Error:                          \t{}".format('|'.join(['{0:>0{width}b}'.format(e, width=linear_code._n) for e in error])))
