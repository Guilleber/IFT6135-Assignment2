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
parser.add_argument('--exp', type=str, default='d',
                    help='s for short experiments and l for long experiments')
parser.add_argument('--train', action='store_true',
                    help='if you want to plot training curves in addition to the validation curves')

args = parser.parse_args()


colors = ["aquamarine","Maroon","Orange","Orchid","Olive","red","Sienna","yellow",
"pink","cyan","black","grey","green","burlywood","fuchsia","purple","plum","teal","tan","Khaki","indigo"]

if args.exp == 's':
    args.results_dir='short_exp'
    e=6
else:
    e=40
    if args.exp == 'd':
        args.results_dir='Default'
    else:
        args.results_dir='long_exp'

#args.results_dir='tmppp'

c=0
i=1
x_max=math.inf


if args.ploty == 'a':
    with open(args.results_dir+'/index'+'_'+args.architecture+'_'+args.exp+'.csv','w') as f:
        f.write("index&model&optimizer&initial\_lr&batch\_size&seq\_len&hidden\_size&num\_layers&dp\_keep\_prob&train\_ppl&best\_val\_ppl\\\\ \n")
        f.write("\\hline \n")
    for (root, dirs, files) in os.walk(args.results_dir):
        table=[]
        for d in dirs:
            if args.architecture in d:
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
                    plt.xlim(0,x_max)
                plt.ylabel("ppls")
                plt.title(args.architecture+" experiments")
                c+=1
                with open(root+'/'+d+"/log.txt",'r') as f:
                    x=f.readlines()
                    best_val_ppl=x[-1].split('\t')[-2].split(': ')[1]
                    train_ppl=x[-1].split('\t')[-4].split(': ')[1]

                s=d.split('=')
                s=s[1:len(s)]
                v=[str(i)]
                for pi,p in enumerate(s):
                    if pi==1:
                        v.append(p.split('_i')[0])
                    else:
                        v.append(p.split('_')[0])
                v.append(train_ppl)
                v.append(best_val_ppl)
                table.append(v)
                i+=1
        table.sort(key=lambda x:x[-1])
        for row in table :
            with open(args.results_dir+'/index'+'_'+args.architecture+'_'+args.exp+'.csv','a') as f:
                f.write(('&').join(row)+'\\\\ \n')

    plt.legend(loc='upper left', prop={'size':5},bbox_to_anchor=(0,1))
    plt.ylim(0,2000)
    plt.savefig(args.results_dir+'/'+args.architecture+'_'+args.plotx+'_'+args.exp+'(zoom)',quality=95,dpi=400)


elif args.ploty == 'o':
    with open(args.results_dir+'/index'+'_'+args.optimizer+'_'+args.exp+'.csv','w') as f:
        f.write("index&model&optimizer&initial\_lr&batch\_size&seq\_len&hidden\_size&num\_layers&dp\_keep\_prob&train\_ppl&best\_val\_ppl\\\\ \n")
        f.write("\\hline \n")
    for (root, dirs, files) in os.walk(args.results_dir):
        table=[]
        for d in dirs:
            if args.optimizer in d :
                if args.optimizer =='SGD' and d.split("=")[0].split('_')[2]!='model':
                    continue
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
                    plt.xlim(0,x_max)
                plt.ylabel("ppls")
                plt.title(args.optimizer+" experiments")
                c+=1
                with open(root+'/'+d+"/log.txt",'r') as f:
                    x=f.readlines()
                    best_val_ppl=x[-1].split('\t')[-2].split(': ')[1]
                    train_ppl=x[-1].split('\t')[-4].split(': ')[1]

                s=d.split('=')
                s=s[1:len(s)]
                v=[str(i)]
                for pi,p in enumerate(s):
                    if pi==1:
                        v.append(p.split('_i')[0])
                    else:
                        v.append(p.split('_')[0])
                v.append(train_ppl)
                v.append(best_val_ppl)
                table.append(v)
                i+=1
        table.sort(key=lambda x:x[-1])
        for row in table :
            with open(args.results_dir+'/index'+'_'+args.optimizer+'_'+args.exp+'.csv','a') as f:
                f.write(('&').join(row)+'\\\\ \n')
    plt.ylim(0,2000)
    plt.legend(loc='upper left', prop={'size':5},bbox_to_anchor=(0,1))
    plt.savefig(args.results_dir+'/'+args.optimizer+'_'+args.plotx+'_'+args.exp+'(zoom)',quality=95,dpi=400)
