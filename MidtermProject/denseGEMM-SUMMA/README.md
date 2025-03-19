Checkout the report for a full breakdown of everything. 

Otherwise, to run the code, just:
make clean
make
mpirun -np 4 ./summa -m 16 -n 16 -k 16 -s c -v -p

the -s flag should either be "c" for stationary-c or "a" for stationary-a

the -b flag doesn't do anything.

Alternatively, run 
make clean
make
make test

to go ahead and run through all of the possible matrix compisitions. 