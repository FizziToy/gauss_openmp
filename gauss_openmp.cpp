#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <fstream>
#include <sstream>

using namespace std;

int main()
{
    int N = 1500;

    vector<vector<double>> A(N, vector<double>(N));
    vector<double> b(N);
    vector<double> x(N);

    srand(time(0));

    for (int i = 0; i < N; i++)
    {
        double sum = 0;

        for (int j = 0; j < N; j++)
        {
            A[i][j] = rand() % 10 + 1;
            sum += abs(A[i][j]);
        }

        A[i][i] = sum + 1;
        b[i] = rand() % 100;
    }

    ofstream logFile("openmp_log.txt");

    logFile << "OpenMP Gauss method log" << endl;
    logFile << "Matrix size: " << N << endl;
    logFile << "Max threads: " << omp_get_max_threads() << endl << endl;

    double start = omp_get_wtime();

#pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        int threadCount = omp_get_num_threads();

#pragma omp single
        {
            logFile << "Parallel region started" << endl;
            logFile << "Number of threads: " << threadCount << endl << endl;
        }

        for (int k = 0; k < N; k++)
        {
#pragma omp single
            {
                if (k < 10)
                {
                    logFile << "----------------------------------------" << endl;
                    logFile << "Step k = " << k << endl;
                    logFile << "Pivot element A[" << k << "][" << k << "] = "
                        << A[k][k] << endl;
                    logFile << "Pivot b[" << k << "] = " << b[k] << endl;
                }
            }

#pragma omp for schedule(static)
            for (int i = k + 1; i < N; i++)
            {
                double oldAik = A[i][k];
                double oldB = b[i];

                double factor = A[i][k] / A[k][k];

                for (int j = k; j < N; j++)
                {
                    A[i][j] -= factor * A[k][j];
                }

                b[i] -= factor * b[k];

                if (k < 10 && i < k + 10)
                {
#pragma omp critical
                    {
                        logFile << "Thread " << threadId
                            << " processed row " << i
                            << ": factor = " << factor
                            << ", A[" << i << "][" << k << "] "
                            << oldAik << " -> " << A[i][k]
                            << ", b[" << i << "] "
                            << oldB << " -> " << b[i]
                            << endl;
                    }
                }
            }

#pragma omp barrier
        }
    }

    for (int i = N - 1; i >= 0; i--)
    {
        x[i] = b[i];

        for (int j = i + 1; j < N; j++)
        {
            x[i] -= A[i][j] * x[j];
        }

        x[i] /= A[i][i];
    }

    double end = omp_get_wtime();

    logFile << endl << "Result vector x:" << endl;

    for (int i = 0; i < 10; i++)
    {
        logFile << "x[" << i << "] = " << x[i] << endl;
    }

    logFile.close();

    cout << "Matrix size: " << N << endl;
    cout << "Threads: " << omp_get_max_threads() << endl;
    cout << "Execution time: " << end - start << " sec" << endl;

    return 0;
}