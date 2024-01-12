import time
import Environment
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

### User specified parameters ###
USE_GPU = True
N_EPISODE = 5000

## 'inc' if increasing the size from the minimum cross-section,
## 'dec' if decreasing the size from the maximum cross-section
MODE = 'inc'
#################################

t1 = time.time()
env = Environment.Environment(gpu=USE_GPU,mode=MODE)
# env.Train(N_EPISODE) # Uncomment when training
t2 = time.time()
env.Test(test_model=1)
print("time: {:.3f} seconds".format(t2-t1))




