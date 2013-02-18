This application takes the Cooley-Tukey Fast Fourier Transform (FFT) algorithm and runs it on a variable number of corse using OpenMP. 

This would take the already efficient O(N Log N) and turn it into O((N Log N) / P); P being the number of cores used during a run. 
