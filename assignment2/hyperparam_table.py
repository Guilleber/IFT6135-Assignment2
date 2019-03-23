import argparse
import os

parser = argparse.ArgumentParser(description='Hyperparameters experiments index table')

parser.add_argument('--results_dir', type=str, default='long_exp',
                    help='dir where models are saved')

parser.add_argument('--exp', type=str, default='s',
                    help='s for short experiments and l for long experiments')

args = parser.parse_args()



if args.exp == 's':
    args.results_dir='short_exp'
    epoch=6


with open(args.results_dir+'/index.csv','w') as f:
    f.write("index,model,optimizer,initial_lr,batch_size,seq_len,hidden_size,num_layers,dp_keep_prob\n")

i=1

for (root, dirs, files) in os.walk(args.results_dir):
    for d in dirs:
        s=d.split('=')
        s=s[1:len(s)]
        v=[]
        for p in s:
            v.append(p.split('_')[0])
        with open(args.results_dir+'/index.csv','a') as f:
            f.write(str(i)+','+(',').join(v)+'\n')
        i+=1
