#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "run-sin.py"])
if r!=0:
 exit(r)

r = call(["python3", "run-sin.py"])
if r!=0:
 exit(r)

r = call(["python3", "run-cmaes.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-cmaes.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-vracer.py"])
if r!=0:
   exit(r)

r = call(["python3", "run-vracer.py"])
if r!=0:
   exit(r)

r = call(["python3", "run-dvracer.py"])
if r!=0:
   exit(r)

r = call(["python3", "run-dvracer.py"])
if r!=0:
   exit(r)

exit(0)
