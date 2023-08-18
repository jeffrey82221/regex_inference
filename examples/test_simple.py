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
    e = Inference(verbose=True, use_openai=False, n_thread=1, model_id='EleutherAI/gpt-neox-20b', max_length=1000)
    start = time.time()
    regex = e.run(patterns)
    end = time.time()
    print('run time =', end - start)
    print(regex)
