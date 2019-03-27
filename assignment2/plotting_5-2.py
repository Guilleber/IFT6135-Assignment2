##############################################################################
#
# PLOTTING THE AVERAGE LOSSES AND GRADENTS WRT HIDDEN STATES PER TIME STEP
#
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')


parser.add_argument('--data', type=str, default='q5',
                    help='location of the data corpus. We suggest you change the default\
                    here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--save_dir', type=str, default='q5',
                    help='path where you want to save the results')
parser.add_argument('--seq_len', type=int, default='35',
                    help='number of timesteps')

args = parser.parse_args()

losses_rnn = np.load(args.data+'/'+'RNN/losses.npy')[()]
losses_rnn =list(map(lambda t: t/max(losses_rnn),losses_rnn))
losses_gru = np.load(args.data+'/'+'GRU/losses.npy')[()]
losses_gru=list(map(lambda t: t/max(losses_gru),losses_gru))
grads_rnn=np.load(args.data+'/'+'RNN/grads.npy')[()]
grads_rnn=list(map(lambda t: t/max(grads_rnn),grads_rnn))
grads_gru=np.load(args.data+'/'+'GRU/grads.npy')[()]
grads_gru=list(map(lambda t: t/max(grads_gru),grads_gru))

plt.figure()
plt.title("Loss L_t as a function of t ")
plt.xlabel("Time step t")
plt.xlim(1, args.seq_len)
plt.ylabel("Loss")
plt.ylim(0, 1)
plt.plot(np.arange(1, args.seq_len + 1), losses_rnn,color='blue',label='rnn')
plt.plot(np.arange(1, args.seq_len + 1), losses_gru,color='green',label='gru')
plt.savefig("{dir}/loss_t.png".format(dir=args.save_dir))


plt.figure()
plt.title("Norm of gradient of L_T wrt h_t as a function of t")
plt.xlabel("Time step t")
plt.xlim(1, args.seq_len)
plt.ylabel("Norm")
plt.ylim(0, 1)
plt.plot(np.arange(1, args.seq_len + 1), grads_rnn,color='blue',label='rnn')
plt.plot(np.arange(1, args.seq_len + 1), grads_gru,color='green',label='gru')
plt.savefig("{dir}/grad_norm.png".format(dir=args.save_dir))
