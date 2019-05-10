# 1mm scale
setscale 1 force
setoption costfunction bbr
setoption optimisationtype brent
setoption tolerance 0.0005 0.0005 0.0005 0.02 0.02 0.02 0.002 0.002 0.002 0.001 0.001 0.001
#setoption tolerance 0.005 0.005 0.005 0.2 0.2 0.2 0.02 0.02 0.02 0.01 0.01 0.01
setoption boundguess 1
setoption bbrstep 200
clear UA
clear UU
clear UV
clear U
setrowqsform UU
setrow UU 1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1
measurecost 6 UU:1-2  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 8
gridmeasurecost 6 UU:1-2  -0.07 0.07 0.07  -0.07 0.07 0.07  -0.07 0.07 0.07  -4.0 4.0 4.0  -4.0 4.0 4.0  -4.0 4.0 4.0  0.0 0.0 0.0  abs 8
sort U
copy U UA
clear U
optimise 6 UA:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 8
setoption optimisationtype powell
optimise 6 U:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 8
setoption optimisationtype brent
optimise 6 U:2  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4
sort U
setoption tolerance 0.0002 0.0002 0.0002 0.02 0.02 0.02 0.002 0.002 0.002 0.001 0.001 0.001
setoption bbrstep 2
clear UU
copy U UU
clear U
gridmeasurecost 6 UU:1  -0.0017 0.0017 0.0017  -0.0017 0.0017 0.0017  -0.0017 0.0017 0.0017  -0.1 0.1 0.1  -0.1 0.1 0.1  -0.1 0.1 0.1  0.0 0.0 0.0  abs 8
sort U
clear UB
copy U UB
clear U
setoption optimisationtype brent
optimise 6 UB:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 8
setoption optimisationtype powell
optimise 12 U:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 8
setoption optimisationtype brent
optimise 12 U:2  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4
sort U
print U:1

