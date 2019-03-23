import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Hyperparameters search')

parser.add_argument('--results_dir', type=str, default='Results',
                    help='dir where models are saved')
parser.add_argument('--architecture', type=str, default='TRANSFORMER',
                    help='architechture to plot')

args = parser.parse_args()

colors = ["aquamarine","Maroon","Light Salmon","Medium Violet Red","Orange Red","Orange","Orchid","Medium Blue","Lime,Medium Purple","Olive","red","Sienna","yellow",
"Hot pink","cyan","black","deep pink","dark grey","Dark Sea Green"]

i=0

for (root, dirs, files) in os.walk(args.results_dir):
	for d in dirs:
		if(d[0]==args.architecture[0]):
			x=np.load(os.path.join(root,d)+'/learning_curves.npy')[()]
			plt.plot(x['clock'],x['val_ppls'],colors[i])
			plt.label(d)
			i+=1			

