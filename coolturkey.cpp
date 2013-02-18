/* 
 * File:   coolturkey.cpp
 * Author: stephen
 *
 * Created on May 4, 2011, 7:49 AM
 */

#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <complex>

using namespace std;

/******************** My FFT Functions ***********************/
void BitReversal(int n, double xReal[], double xImg[], double log2_of_n);
void FillArray(int n, double xReal[], double xImg[], double yReal[], double yImg[]);
void CalcFFT(int log2_of_n, double xReal[], double xImg[], double wReal[], double wImg[], int N);
timespec GetTimeElapsed(timespec s, timespec f);
void GetTwiddleFactors(int n, int N, double wReal[], double wImg[]);
/*************************************************************/

/*****************  Oracle Functions   ***********************/
static int bitrev(int inp, int numbits);
static void compute_W(int n, double W_re[], double W_im[]);
static void fft(int n, double A_re[], double A_im[], double W_re[], double W_im[]);
static int log_2(int n);
static void permute_bitrev(int n, double A_re[], double A_im[]);
/*************************************************************/

/*
 *  Caculates FFT of array, with error check and timing, compared to oracle fft function
 */
int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cerr << "usage: FFT <size of array> <number of threads>" << endl;
        exit(1);
    }

    int n = atoi(argv[1]); //Assign size of array
    if (!(n > 0) || !((n & (n - 1)) == 0)) //Test if
    {
        cerr << "usage: <size of array> must be power of 2" << endl;
        exit(1);
    }

    int p = atoi(argv[2]); //Assign number of threads to use
    double log2_of_n = log10(n) / log10(2); //Set for use later
    double *xReal = new double[n];
    double *xImg = new double[n];
    double *yReal = new double[n];
    double *yImg = new double[n];
    double *wReal = new double[(int) log2_of_n];
    double *wImg = new double[(int) log2_of_n];
    double *W_OracleR = new double[n / 2];
    double *W_OracleI = new double[n / 2];
    //cout << "Number of Cores: " << omp_get_num_procs() << endl;
    //cout << "Max Number of Threads: " << omp_get_max_threads() << endl;
    omp_set_num_threads(p); //Set how many proc's to user
//    #pragma omp parallel
//    {
//        #pragma omp master
//        {
//            int num = omp_get_num_threads(); //Get number of threads for master thread
//            cout << "Number of Threads: " << num << endl;
//        }
//        #pragma omp barrier
//    }

    FillArray(n, xReal, xImg, yReal, yImg);
    timespec sTime;
    clock_gettime(CLOCK_MONOTONIC, &sTime); //Set start time
    BitReversal(n, xReal, xImg, log2_of_n);
    GetTwiddleFactors(log2_of_n, n, wReal, wImg);
    CalcFFT(log2_of_n, xReal, xImg, wReal, wImg, n);
    timespec fTime;
    clock_gettime(CLOCK_MONOTONIC, &fTime); //Set end time
    double frac = 1.0 / (double) n; //Set fraction var
    double error = 0.0;
    for (int i = 0; i < n; i++)
    {   //Iterate through values and compile error with FFT ( FFT ( X(1:N) ) ) == N * X(1:N)
        //Thus we want close to zero as possible
        error = error + pow(yReal[i] - frac * xReal[i], 2) + pow(yImg[i] - frac * xImg[i], 2);
    }
    error = sqrt(frac * error); //Calculate final error
    cout << "Parallel FFT Time-elapsed: " << GetTimeElapsed(sTime, fTime).tv_sec
            << "." << GetTimeElapsed(sTime, fTime).tv_nsec << " seconds; ";// << endl;
    cout << "Error (Mine): " << error << endl;

    //Oracle Code
//    FillArray(n, xReal, xImg, yReal, yImg);
//    compute_W(n, W_OracleR, W_OracleI);
//    fft(n, xReal, xReal, W_OracleR, W_OracleI);
//    permute_bitrev(n, xReal, xImg);
//
//    frac = 1.0 / (double) n;
//    error = 0.0;
//    for (int i = 0; i < n; i = i++)
//    {
//        error = error + pow(yReal[i] - frac * xReal[i], 2) + pow(yImg[i] - frac * xImg[i], 2);
//    }
//    error = sqrt(frac * error);
//    cout << "Error (Oracle): " << error << endl << endl;
    return 0;
}

/******************** My FFT Functions ***********************/
/**********************************************************/ 
//   BitReversal
//   Description: Bitreverses the arrays to prepare it for use in FFT
//   Preconditions: n = size of arrays
//                  xReal = real number array
//                  xImg = imaginary number array
//                  log2_of_n = the log of n to the base 2
//   Postconditions: Returns correctly position array
/**********************************************************/
void BitReversal(int n, double xReal[], double xImg[], double log2_of_n)
{
    int i;
    double temp;
    #pragma omp parallel for default(none) private(i, temp) shared(n, xReal, xImg, log2_of_n)
    for (i = 1; i < n; i++)
    {
        int k, rev = 0;
        int inp = i;
        for (k = 0; k < log2_of_n; k++)
        {
            rev = (rev << 1) | (inp & 1);
            inp >>= 1;
        }

        if (rev <= i) continue; //Skip if already done
        temp = xReal[i]; //Store into temp values and swap
        xReal[i] = xReal[rev];
        xReal[rev] = temp;
        temp = xImg[i];
        xImg[i] = xImg[rev];
        xImg[rev] = temp;
    }
}

/**********************************************************/
//   CalcFFT
//   Description: Calculates the Fast Fourier Transform through butterfly operations
//   Preconditions: log2_of_n = the log of n to the base 2
//                  xReal = real number array
//                  xImg = imaginary number array
//                  wReal = real number twiddle factor array
//                  wImg = imaginary number twiddle factor array
//                  N = size of arrays
//   Postconditions: Returns the Fast Fourier Transforms of xReal and xImg
/**********************************************************/
void CalcFFT(int log2_of_n, double xReal[], double xImg[], double wReal[], double wImg[], int N)
{
    int n, d, i, k;
    double temp_w, temp_x;
    for (n = 1; n < log2_of_n + 1; n++)
    {   //Iterate through time slice
        d = pow(2, n);
        #pragma omp parallel for default(none) private(k, i, temp_w, temp_x) shared(n, N, d, xReal, xImg, wReal, wImg, log2_of_n)
        for (k = 0; k < (d / 2); k++)
        {   //Iterate through even and odd elements
            for (i = k; i < N; i += d)
            { //Butterfly operation
                temp_w = wReal[n - 1] * xReal[i + (d / 2)]; //Multiply by twiddle factor
                temp_x = xReal[i];  //Store in temp's and restore with correct values
                xReal[i] = temp_w + temp_x;
                xReal[i + (d / 2)] = temp_x - temp_w;
                temp_w = wImg[n - 1] * xImg[i + (d / 2)]; //Multiply by twiddle factor
                temp_x = xImg[i]; //Store in temp's and restore with correct values
                xImg[i] = temp_w + temp_x;
                xImg[i + (d / 2)] = temp_x - temp_w;
            }
        }
    }
}

/**********************************************************/
//   FillArray
//   Description: Fills arrays with random values
//   Preconditions: n = size of arrays
//                  xReal = real number array
//                  xImg = imaginary number array
//                  xReal = copy of real number array for error checking
//                  xImg = copy of imaginary number array for error checking
//   Postconditions: Returns arrays filled with random values
/**********************************************************/
void FillArray(int n, double xReal[], double xImg[], double yReal[], double yImg[])
{
    int i;
    struct drand48_data buffer;
    srand48_r(time(NULL) ^ omp_get_thread_num(), &buffer); //Seed for each thread
    //#pragma omp parallel for default(none) private(i) shared(x,n)
    for (i = 0; i < n; i++)
    {
        drand48_r(&buffer, &xReal[i]); //Store random double into xReal
        yReal[i] = xReal[i]; //Copy
        drand48_r(&buffer, &xImg[i]); //Store random double into xImg
        yImg[i] = xImg[i]; //Copy
    }
}

/**********************************************************/ 
//   GetTimeElapsed
//   Description: Calculates correct time in proper format
//   Preconditions: s = start time
//                  f = finish time
//   Postconditions: Returns timespec struct with elapsed time
/**********************************************************/
timespec GetTimeElapsed(timespec s, timespec f)
{
    timespec totTime;
    if ((f.tv_nsec - s.tv_nsec) < 0) //so we do not return a negative incorrect value
    {
        totTime.tv_sec = f.tv_sec - s.tv_sec - 1;
        totTime.tv_nsec = 1000000000 + f.tv_nsec - s.tv_nsec;
    }
    else
    {
        totTime.tv_sec = f.tv_sec - s.tv_sec;
        totTime.tv_nsec = f.tv_nsec - s.tv_nsec;
    }
    return totTime;
}

/**********************************************************/ 
//   GetTwiddleFactors
//   Description: Calculates the twiddle factors before the FFT function
//   Preconditions: log2_of_n = the log of n to the base 2
//                  wReal = real number twiddle factor array
//                  wImg = imaginary number twiddle factor array
//   Postconditions: Returns wReal and wImg with calculated values
/**********************************************************/
void GetTwiddleFactors(int log2_of_n, int n, double wReal[], double wImg[])
{
    int i;
#pragma omp parallel for default(none) private(i) shared(n, wReal, wImg, log2_of_n)
    for (i = 0; i < log2_of_n; i++)
    {
        wReal[i] = cos(((double) i * 2.0 * M_PI) / ((double) n));
        wImg[i] = sin(((double) i * 2.0 * M_PI) / ((double) n));
    }
}
/*************************************************************/

/*****************  Oracle Functions   ***********************/

static int bitrev(int inp, int numbits)
{
    int i, rev = 0;
    for (i = 0; i < numbits; i++)
    {
        rev = (rev << 1) | (inp & 1);
        inp >>= 1;
    }
    return rev;
}

static void compute_W(int n, double W_re[], double W_im[])
{
    int i, br;
    int log2n = log_2(n);
    for (i = 0; i < (n / 2); i++)
    {
        br = bitrev(i, log2n - 1);
        W_re[br] = cos(((double) i * 2.0 * M_PI) / ((double) n));
        W_im[br] = sin(((double) i * 2.0 * M_PI) / ((double) n));
        //cout << "Twiddle: " << W_re[br] << endl;
    }
}

static void fft(int n, double A_re[], double A_im[], double W_re[], double W_im[])
{
    double w_re, w_im, u_re, u_im, t_re, t_im;
    int m, g, b;
    int mt, k;

    /* for each stage */
    for (m = n; m >= 2; m >>= 1)
    {
        mt = m >> 1;

        /* for each group of butterfly */
        for (g = 0, k = 0; g < n; g += m, k++)
        {
            /* each butterfly group uses only one root of
             * unity. actually, it is the bitrev of this
             * group's number k. BUT 'bitrev' it as a
             * log2n-1 bit number because we are using a
             * lookup array of nth root of unity and using
             * cancellation lemma to scale nth root to
             * n/2, n/4,... th root.
             *
             * It turns out like the foll.
             *   w.re = W[bitrev(k, log2n-1)].re;
             *   w.im = W[bitrev(k, log2n-1)].im;
             *
             * Still, we just use k, because the lookup
             * array itself is bit-reversal permuted.
             */
            w_re = W_re[k];
            w_im = W_im[k];

            /* for each butterfly */
            for (b = g; b < (g + mt); b++)
            {
                t_re = w_re * A_re[b + mt] - w_im * A_im[b + mt];
                t_im = w_re * A_im[b + mt] + w_im * A_re[b + mt];

                u_re = A_re[b];
                u_im = A_im[b];
                A_re[b] = u_re + t_re;
                A_im[b] = u_im + t_im;
                A_re[b + mt] = u_re - t_re;
                A_im[b + mt] = u_im - t_im;

            }
        }
    }
}

static int log_2(int n)
{
    int res;
    for (res = 0; n >= 2; res++)
        n = n >> 1;
    return res;
}

static void permute_bitrev(int n, double A_re[], double A_im[])
{
    int i, bri, log2n;
    double t_re, t_im;

    log2n = log_2(n);

    for (i = 0; i < n; i++)
    {
        bri = bitrev(i, log2n);

        /* skip already swapped elements */
        if (bri <= i) continue;

        t_re = A_re[i];
        t_im = A_im[i];
        A_re[i] = A_re[bri];
        A_im[i] = A_im[bri];
        A_re[bri] = t_re;
        A_im[bri] = t_im;
    }
}
/*************************************************************/