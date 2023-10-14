from regex_inference import Inference
import time

patterns = [str(i) for i in range(10)]
if __name__ == '__main__':
    e = Inference(verbose=True, max_iteration=3)
    start = time.time()
    regex = e.run(patterns, n_fold=1, train_rate=0.1)
    end = time.time()
    print('run time =', end - start)
    print(regex)
    print(e.openai_summary)
