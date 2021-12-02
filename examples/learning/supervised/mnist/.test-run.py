#! /usr/bin/env python3
from subprocess import call

r = call(["bash", "get_mnist.sh"])
if r!=0:
  exit(r)

r = call(["python3", "run-mnist-autoencoder.py", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "run-mnist-classifier.py", "--test"])
if r!=0:
  exit(r)

exit(0)
