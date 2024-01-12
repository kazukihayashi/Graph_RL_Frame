import time
import Environment
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="Specify inc if increasing the size from the minimum cross-sections; specify dec if reducing the size from the maximum cross-sections.", default="inc", type=str)
parser.add_argument("--test_model", help="Structural model to test the trained model's performance", default=1, type=int)
parser.add_argument("--train", help="True if implement the training. False if only using the pre-trained machine learning model.", action='store_true')
parser.add_argument("--use_gpu", help="Use GPU if True, use CPU if False", action='store_true')
parser.add_argument("--n_episode", help="Number of episodes to train the machine learning model.", default=5000, type=int)
args = parser.parse_args()

t1 = time.time()
env = Environment.Environment(gpu=args.use_gpu,mode=args.mode)
if args.train:
    env.Train(args.n_episode)
t2 = time.time()
env.Test(test_model=1)
print("time: {:.3f} seconds".format(t2-t1))




