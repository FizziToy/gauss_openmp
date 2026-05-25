#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

int main()
{
    int N = 3000;

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

    double start = omp_get_wtime();

    for (int k = 0; k < N; k++)
    {
#pragma omp parallel for
        for (int i = k + 1; i < N; i++)
        {
            double factor = A[i][k] / A[k][k];

            for (int j = k; j < N; j++)
            {
                A[i][j] -= factor * A[k][j];
            }

            b[i] -= factor * b[k];
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

    cout << "Matrix size: " << N << endl;
    cout << "Processes: " << omp_get_max_threads() << endl;
    cout << "Execution time: " << end - start << " sec" << endl;

    return 0;
}