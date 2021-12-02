//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

std::string OPTIONS = "-bpdx 4 -bpdy 2 -levelMax 7 -levelStart 4 -Rtol 2 -Ctol 1 -extent 2 -CFL 0.4 -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -maxPoissonRestarts 10 -maxPoissonIterations 10000 -bAdaptChiGradient 0 -tdump 0.1 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

std::string OBJECTShalfDisk = "halfDisk radius=0.06 angle=20 xpos=0.6 bForced=1 bFixed=1 xvel=0.15 tAccel=5 \n\
stefanfish L=0.2 T=1 xpos=0.9";

std::string OBJECTSnaca = "NACA L=0.12 xpos=0.6 angle=0 fixedCenterDist=0.299412 bFixed=1 xvel=0.15 Apitch=13.15 Fpitch=0.715 tAccel=5 \n\
stefanfish L=0.2 T=1 xpos=0.9";

std::string OBJECTSstefanfish = "stefanfish L=0.2 T=1 xpos=0.6 bFixed=1 pid=1 \n\
stefanfish L=0.2 T=1 xpos=0.9";

std::string OBJECTSwaterturbine = "waterturbine semiAxisX=0.05 semiAxisY=0.017 xpos=0.6 bForced=1 bFixed=1 xvel=0.15 angvel=0.79 tAccel=0 \n\
stefanfish L=0.2 T=1 xpos=0.9";
