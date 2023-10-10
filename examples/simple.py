from regex_inference import Inference
import time

patterns = [
    "0",
    "9",
    "",
    "123",
    "apple",
    "",
    "@",
    "中華文化",
    "   "
]

if __name__ == '__main__':
    e = Inference(verbose=False)
    start = time.time()
    regex = e.run(patterns, n_fold=1, train_rate=0.5)
    end = time.time()
    print('run time =', end - start)
    print(regex)
