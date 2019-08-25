#!/usr/bin/python3

import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--server', "-s", type=str, default="kappa", 
		help='Servername')
args = parser.parse_args()
		
target = "~/cotk_contrib"
exclude = [".git", "__pycache__", "OpenSubtitles", "glove", "cotk_contrib.egg-info", "cache", "wordvec", "log", "tensorboard", "model", "dataset", "data", "output", "env"]

cmd = ["rsync -vrz --delete "]
for i in exclude:
	cmd.append(" --exclude ")
	cmd.append('"%s"' % i)
this_path = os.path.dirname(__file__)
if this_path == "":
	this_path = "."
cmd.append(" %s/ %s:%s" % (this_path, args.server, target))
print("".join(cmd))
os.system("".join(cmd))
