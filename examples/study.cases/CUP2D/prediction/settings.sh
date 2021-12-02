#!/bin/bash
# Defaults for Options
BPDX=${BPDX:-32}
BPDY=${BPDY:-16}
NAGENTS=$(($(($(($BPDX*$BPDY))-$((2*$BPDX))))-$((2*$(($BPDY-2))))))
LEVELS=${LEVELS:-1}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.2}
PT=${PT:-1e-8}
PTR=${PTR:-0}
# Defaults for Objects
XPOS=${XPOS:-0.2}
XVEL=${XVEL:-0.15}
RADIUS=${RADIUS:-0.0375}
NU=${NU:-0.0001125} # Re=100 <-> NU=0.0001125
OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0 -nu $NU -tend 0 -muteAll 0 -verbose 0 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1"
OBJECTS="disk radius=$RADIUS xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=2"