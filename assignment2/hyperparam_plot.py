import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import math


parser = argparse.ArgumentParser(description='Hyperparameters search plotting')

parser.add_argument('--results_dir', type=str, default='long_exp',
                    help='dir where models are saved')
parser.add_argument('--architecture', type=str, default='TRANSFORMER',
                    help='architechture to plot')
parser.add_argument('--optimizer', type=str, default='ADAM',
                    help='optimizer to plot')
parser.add_argument('--plotx', type=str, default='epoch',
                    help='epoch for plotting over epochs and clock for plotting over wall clock time')
parser.add_argument('--ploty', type=str, default='o',
                    help='o for plotting optimizer a for architecture')
parser.add_argument('--exp', type=str, default='s',
                    help='s for short experiments and l for long experiments')
parser.add_argument('--train', action='store_true',
                    help='if you want to plot training curves in addition to the validation curves')

args = parser.parse_args()


colors = ["aquamarine","Maroon","Orange","Orchid","Olive","red","Sienna","yellow",
"pink","cyan","black","grey","green","burlywood","fuchsia"]

if args.exp == 's':
    args.results_dir='short_exp'
    e=6
else:
    e=40

c=0
i=1
x_max=math.inf

if args.ploty == 'a':
    for (root, dirs, files) in os.walk(args.results_dir):
        for d in dirs:
            if d.split('_')[0]==args.architecture:
                x=np.load(os.path.join(root,d)+'/learning_curves.npy')[()]
                if args.plotx=='epoch':
                    plt.plot(range(e),x['val_ppls'],color=colors[c],label=str(i)+'_val')
                    if args.train:
                        plt.plot(range(e),x['train_ppls'],color=colors[c],label=str(i)+'_train',linestyle='--')
                    plt.xlabel("epoch")
                elif args.plotx=='clock':
                    time=list(map(lambda t: t-x['clock'][0],x['clock']))
                    plt.plot(time,x['val_ppls'],color=colors[c],label=str(i)+'_val')
                    if args.train:
                        plt.plot(time,x['train_ppls'],color=colors[c],label=str(i)+'_train',linestyle='--')
                    if time[e-1]<x_max:
                        x_max=time[e-1]
                    plt.xlabel("wall_clock_time")
                plt.ylabel("ppls")
                plt.xlim(0,x_max)
                plt.title(args.architecture+" experiments")
                c+=1
            i+=1

    plt.legend(loc='upper left', prop={'size':6},bbox_to_anchor=(0,1))
    plt.savefig(args.results_dir+'/'+args.architecture+'_'+args.plotx)


elif args.ploty == 'o':
    for (root, dirs, files) in os.walk(args.results_dir):
        for d in dirs:
            if d.split('_')[1]==args.optimizer:
                x=np.load(os.path.join(root,d)+'/learning_curves.npy')[()]
                if args.plotx=='epoch':
                    plt.plot(range(e),x['val_ppls'],color=colors[c],label=str(i)+'_val')
                    if args.train:
                        plt.plot(range(e),x['train_ppls'],color=colors[c],label=str(i)+'_train',linestyle='--')
                    plt.xlabel("epoch")
                elif args.plotx=='clock':
                    time=list(map(lambda t: t-x['clock'][0],x['clock']))
                    plt.plot(time,x['val_ppls'],color=colors[c],label=str(i)+'_val')
                    if args.train:
                        plt.plot(time,x['train_ppls'],color=colors[c],label=str(i)+'_train',linestyle='--')
                    if time[e-1]<x_max:
                        x_max=time[e-1]
                    plt.xlabel("wall_clock_time")
                plt.ylabel("ppls")
                plt.xlim(0,x_max)
                plt.title(args.optimizer+" experiments")
                c+=1
            i+=1

    plt.legend(loc='upper left', prop={'size':6},bbox_to_anchor=(0,1))
    plt.savefig(args.results_dir+'/'+args.optimizer+'_'+args.plotx)
