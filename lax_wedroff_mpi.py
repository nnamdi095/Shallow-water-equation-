from mpi4py import MPI  
import time
import numpy as np



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start = time.time()


nxt = 41   
nt = 500
X  = 1.0     
T = 0.4   
g = 9.8
pi = np.pi 


dt = T/nt
dx = X/(nxt-1)
landa = dt/dx



# Work Division - sharing the x partitions amongs the processors
def local_nx(rank, size):
    if rank <= (nxt-1) % size:
        nxp = (nxt-1)/size + 1
    else:
        nxp = (nxt-1)/size
    return nxp

nx = local_nx(rank, size)


#Create a starting data point for each processor
def first_xpoint(rank, size, nxt):
    i_first = 1 + rank*((nxt-1)/size) + min(rank, nxt%size)
    return i_first

i_start = first_xpoint(rank, size, nxt)

#Create an end data point for each processor
def last_xpoint(rank, size, nxt):
    return i_start + nx - 1

i_last = last_xpoint(rank, size, nxt)


# Create the function to compute initial condition at each data points
def f(x):
    return 2 + np.cos(2*pi*x)

#To get the data points for each subset of x for each processor
xp = []
for i in range(i_start, i_last+1):
    xp.append(i)

#Create 1-D array to use in computing h and uh at the x data points
def length_array():
    if rank !=0:
        height_array = np.zeros(nx+1)
        velocity_array = np.zeros(nx+1)
    else:
        height_array= np.zeros(nx+2)
        velocity_array = np.zeros(nx+2)
    return height_array, velocity_array

h, uh = length_array()

#Compute initial condition at the x data points
if rank != 0:
    h[0:nx] = f(np.array(xp)*dx)
    uh[:] = 0
else:
    h[1:nx+1] = f(np.array(xp)*dx)
    uh[:] = 0
    
#Create a function for Boundary condition
def boundary_condition(h, uh, rank):
    if rank == 0:
        h[0] = h[1]
        uh[0] = uh[1]
    	
    if rank == (size-1):
    	h[nx-1] = h[nx-2]
    	uh[nx-1] = uh[nx-2]
    return h, uh

#Apply boundary condition on the initial h and uh
h, uh = boundary_condition(h, uh, rank)

#Transfer h[0] to each left processor to be used in computation of ht and uht
#All processor sends save p0; all processor receives save p(last)
if rank != 0:
	comm.send(h[0], dest = rank-1)
if rank != (size-1):
    if rank == 0:
        h[nx+1] = comm.recv(source = rank+1)
    else:
        h[nx] = comm.recv(source = rank+1)

   
#Create ht and uht arrays
ht, uht = length_array()

#Create uhnew and hnew arrays
h_new, uh_new = length_array()

#Create nx * nt arrays to store values
if rank !=0:
    hp_array = np.zeros((nx, nt+1))
    uhp_array = np.zeros((nx, nt+1))
else:
    hp_array = np.zeros((nx+1, nt+1))
    uhp_array = np.zeros((nx+1, nt+1))
    

    
for j in range(1, nt+1):
    if rank != (size-1) and rank != 0:
        for i in range(1, nx+1):
            ht[i] = 0.5*(h[i-1] + h[i]) - 0.5*landa*(uh[i] - uh[i-1])
        for i in range(1, nx+1):
            uht[i] = 0.5*(uh[i-1] + uh[i]) - 0.5*landa*(uh[i]**2/h[i] + 0.5*g*h[i]**2 - uh[i-1]**2/h[i-1] - 0.5*g*h[i-1]**2)
            
    elif rank == (size-1):
        for i in range(1, nx):
            ht[i] = 0.5*(h[i-1] + h[i]) - 0.5*landa*(uh[i] - uh[i-1])
        for i in range(1, nx):
            uht[i] = 0.5*(uh[i-1] + uh[i]) - 0.5*landa*(uh[i]**2/h[i] + 0.5*g*h[i]**2 - uh[i-1]**2/h[i-1] - 0.5*g*h[i-1]**2)

    else:
        for i in range(0, nx+1):
            ht[i] = 0.5*(h[i] + h[i+1]) - 0.5*landa*(uh[i+1] - uh[i])
        for i in range(0, nx+1):
            uht[i] = 0.5*(uh[i] + uh[i+1]) - 0.5*landa*(uh[i+1]**2/h[i+1] + 0.5*g*h[i+1]**2 - uh[i]**2/h[i] - 0.5*g*h[i]**2)

#Do data transfer - fill ur right neighbour with ur ht[nx] or ht[nx+1] for p0
    if rank != (size-1):
        comm.send(ht[nx], dest = rank+1)
        comm.send(uht[nx], dest = rank+1)
    if rank != 0:
       ht[0] = comm.recv(source = rank-1)
       uht[0] = comm.recv(source = rank-1)

#Compute hnew
       
    if rank != (size-1) and rank != 0:
       for i in range(1, nx+1):
           h_new[i-1]= h[i-1] - landa * (uht[i] - uht[i-1])
           uh_new[i-1] = uh[i-1] - landa*(uht[i]**2/ht[i] + 0.5*g*ht[i]**2 - uht[i-1]**2/ht[i-1] - 0.5*g*ht[i-1]**2)
    
    elif rank == (size-1):
        for i in range(1, nx):
            h_new[i-1]= h[i-1] - landa * (uht[i] - uht[i-1])
            uh_new[i-1] = uh[i-1] - landa*(uht[i]**2/ht[i] + 0.5*g*ht[i]**2 - uht[i-1]**2/ht[i-1] - 0.5*g*ht[i-1]**2)   
    else:
        for i in range(1, nx+1):
            h_new[i]= h[i] - landa * (uht[i] - uht[i-1])
            uh_new[i] = uh[i] - landa*(uht[i]**2/ht[i] + 0.5*g*ht[i]**2    \
                  - uht[i-1]**2/ht[i-1] - 0.5*g*ht[i-1]**2)
            
    h_new, uh_new = boundary_condition(h_new, uh_new, rank)
    
    if rank !=0:    
        for i in range(0, nx+1):
            h[0:nx] = h_new[0:nx]
            uh[0:nx] = uh_new[0:nx]
    else:
        h[0:nx+1] = h_new[0:nx+1]
        uh[0:nx+1] = uh_new[0:nx+1]
#another transfer of h in the form of nhew        
    if rank != 0:
        comm.send(h[0], dest = rank-1)
        comm.send(uh[0], dest = rank-1)
    if rank != (size-1):
        if rank == 0:
            h[nx+1] = comm.recv(source = rank+1)
            uh[nx+1] = comm.recv(source = rank+1)
        else:
            h[nx] = comm.recv(source = rank+1)
            uh[nx] = comm.recv(source = rank+1)
    
    if rank !=0:
        for i in range(0, nx):
            hp_array[0:nx,j] = h[0:nx]
            uhp_array[0:nx,j] = uh[0:nx]
    else:
        for i in range(0, nx+1):
            hp_array[0:nx+1,j] = h[0:nx+1]
            uhp_array[0:nx+1,j] = uh[0:nx+1]

#Sending the final h solution to p0 
if rank != 0:
     comm.send(hp_array, dest = 0)
else:
    for source in range(1, size):
        solution = comm.recv(source = source)
        print("Process 0 receives height array from process ", source, ":", solution)

#Sending final uh solution to p0
if rank != 0:
     comm.send(uhp_array, dest = 0)
else:
    for source in range(1, size):
        solution2 = comm.recv(source = source)
        print("Process 0 receives mass velocity array from process ", source, ":", solution2)
    
end = time.time()
if rank == 0:
    print('Execution Time: ' + str(end-start)) 
        

    
