"""
Evaluate Inference using Precision / Recall / F1 Given Different Train count

https://github.com/shibing624/addressparser/blob/master/tests/addr.csv

TODO: 
- [ ] Add setting of cross validation to help enhance performance for many fold.
- [ ] Add cache for same input to infer_by_fado
- [ ] Enable multi-processing for infer_by_fado
"""
from regex_inference import Evaluator, Inference
import random

def do_experiment(data_path, train_cnt, engine, n_thread):
    whole_patterns = []
    with open(data_path, 'r') as f:
        whole_patterns = f.read().split('\n')
    train_patterns = random.sample(whole_patterns, train_cnt)
    eval_patterns = list(set(whole_patterns) - set(train_patterns))
    print(f'{engine} engine with {n_thread} thread on dataset {data_path}')
    inferencer = Inference(verbose=False, n_thread=n_thread, engine=engine, max_iteration=9)
    regex = inferencer.run(
        train_patterns[:len(train_patterns)//2], val_patterns=train_patterns[len(train_patterns)//2:])
    precision, recall, f1 = Evaluator.evaluate(
        regex, eval_patterns)
    print('f1:', f1)

if __name__ == '__main__':
    do_experiment('data/version.txt', 100, 'ai', 9)
    do_experiment('data/version.txt', 100, 'fado+ai', 9)
    do_experiment('data/address.txt', 100, 'ai', 9)
    do_experiment('data/address.txt', 100, 'fado+ai', 9)
    

"""
#####################################################
# Experiment 1: infer with different training count #
#####################################################

Result:

TRAIN_CNT = 1

regex_list count: 1
precision: 1.0
recall: 0.12842978384465378
f1: 0.2276256541316782

TRAIN_CNT = 10

regex_list count: 2
precision: 0.5001064735945485
recall: 0.5446978021978022
f1: 0.5214505785310993

TRAIN_CNT = 100

regex_list count: 5
precision: 0.4718731351244318
recall: 0.9224456072707243
f1: 0.6243583872893784

TRAIN_CNT = 200

regex_list count: 4
precision: 0.6253244357827017
recall: 0.9697045015189174
f1: 0.7603372028022296

###################################################
# Experiment 2: Compare FAdo and Pure AI approach #
###################################################

# version.txt

TRAIN_CNT = 1

FAdoEngine with 1 thread
f1: 0.2814594543566506
FAdoEngine with 3 thread
f1: 0.5580022971206781
Pure AI Engine with 1 thread
f1: 0.5580022971206781
Pure AI Engine with 3 thread
f1: 0.5580022971206781


TRAIN_CNT = 10

version.txt

FAdoEngine with 1 thread
f1: 0.01429272816540287
FAdoEngine with 3 threadx
f1: 0.8529652738387848
Pure AI Engine with 1 thread
f1: 0.8924556145502123
Pure AI Engine with 3 thread
f1: 0.8319668843358416

# address.txt


TRAIN_CNT = 1

FAdoEngine with 1 thread
f1: 0.1197429906542056
FAdoEngine with 3 thread
f1: 0.06782713085234093
Pure AI Engine with 1 thread
f1: 0.17672047578589636
Pure AI Engine with 3 thread
f1: 0.0782089552238806

TRAIN_CNT = 10


FAdoEngine with 1 thread
f1: 0.3202511773940345
FAdoEngine with 3 thread
f1: 0.741669933359467
Pure AI Engine with 1 thread
f1: 0.9646831156265119
Pure AI Engine with 3 thread
f1: 0.8863833477883782

#####################################################
# Experiment 3: Using Half of training for Validate #
# takes: version.txt / address.txt                  #
# setting: fado+ai                                  #
#####################################################

# version.txt


TRAIN_CNT = 10

FAdoEngine with 1 thread
f1: 0.781630740393627
FAdoEngine with 3 thread
f1: 0.8046762117430709
FAdoEngine with 9 thread
f1: 0.8068507743997377

TRAIN_CNT = 25


FAdoEngine with 1 thread
f1: 0.7171215005993935
FAdoEngine with 3 thread
f1: 0.8406882268599649
FAdoEngine with 9 thread
f1: 0.938113791241664

TRAIN_CNT = 50


FAdoEngine with 1 thread
f1: 0.9015769439912997
FAdoEngine with 3 thread
f1: 0.8693678286016356
FAdoEngine with 9 thread
f1: 0.9373547849721626

TRAIN_CNT = 75

FAdoEngine with 1 thread
f1: 0.9445017356824356
FAdoEngine with 3 thread
f1: 0.7772374686611364
FAdoEngine with 9 thread
f1: 0.9676676989516152

TRAIN_CNT = 100

FAdoEngine with 1 thread
f1: 0.9338618512809219
FAdoEngine with 3 thread
f1: 0.8852559674572108
FAdoEngine with 9 thread
f1: 0.9806002414441731

# address.txt 

TRAIN_CNT = 10
FAdoEngine with 1 thread
f1: 0.0006228589224540642
FAdoEngine with 3 thread
f1: 0.3474903474903475
FAdoEngine with 9 thread
f1: 0.41285537700865266

TRAIN_CNT = 25

FAdoEngine with 1 thread
f1: 0.08338332333533294
FAdoEngine with 3 thread
f1: 0.12444966245964192
FAdoEngine with 9 thread
f1: 0.8865656037637218

TRAIN_CNT = 50

FAdoEngine with 1 thread
f1: 0.6522108843537415
FAdoEngine with 3 thread
f1: 0.8407387090875845
FAdoEngine with 9 thread
f1: 0.7717938783417281

TRAIN_CNT = 75

FAdoEngine with 1 thread
f1: 0.6631243358129649
FAdoEngine with 3 thread
f1: 0.5698044565711687
FAdoEngine with 9 thread
f1: 0.8892812996644889

TRAIN_CNT = 100

FAdoEngine with 1 thread
f1: 0.8091603053435115
FAdoEngine with 3 thread
f1: 0.34174860483656655
FAdoEngine with 9 thread
f1: 0.657487091222031


#######################################################################################
# Experiment 4: Compare FAdo and Pure AI Approach on high train count and thread cnt  #
#######################################################################################

ai engine with 9 thread on dataset data/version.txt
f1: 0.9708049886621315
fado+ai engine with 9 thread on dataset data/version.txt
f1: 0.8225756765082609

ai engine with 9 thread on dataset data/address.txt
f1: 0.9975903614457832
fado+ai engine with 9 thread on dataset data/address.txt
f1: 0.8664850136239781

"""
