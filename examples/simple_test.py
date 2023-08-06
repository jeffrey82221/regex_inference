from regex_inference import Engine
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
    e = Engine()
    start = time.time()
    regex = e.run(patterns)
    end = time.time()
    print('run time =', end - start)
    print(regex)
    e.explain(regex)
