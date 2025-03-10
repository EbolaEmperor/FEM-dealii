#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_relaxation.h>

#include <fstream>

struct CPUTimer
{
  using HRC = std::chrono::high_resolution_clock;
  std::chrono::time_point<HRC>  start;
  CPUTimer() { reset(); }
  void reset() { start = HRC::now(); }
  double operator() () const {
    std::chrono::duration<double> e = HRC::now() - start;
    return e.count();
  }
};

using namespace dealii;

const int N = 1<<15;
const int n_dof = N+1;
SparsityPattern sparsity_pattern;
SparseMatrix<double> system_matrix;
Vector<double> rhs;
Vector<double> solution;


double u(const double &x)
{
  return cos(M_PI * x);
}


double f(const double &x)
{
  return (1. + M_PI*M_PI) * cos(M_PI * x);
}


void make_sparse_pattern()
{
  DynamicSparsityPattern dynamic_sparsity_pattern(n_dof, n_dof);
  for(int k = 0; k <= N; k++)
  {
    dynamic_sparsity_pattern.add(k, k);
    if(k<N) dynamic_sparsity_pattern.add(k, k+1);
    if(k>0) dynamic_sparsity_pattern.add(k, k-1);
  }
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);
  rhs.reinit(n_dof);
  solution.reinit(n_dof);
}


void assemble_system()
{
  for(int k = 0; k <= N; k++)
  {
    int fg = (k==0 || k==N) ? 1 : 2;
    system_matrix.add(k, k, fg*N + fg/(3.*N));
    if(k<N)
    {
      system_matrix.add(k, k+1, -N + 1./(6.*N));
      rhs(k) += .5 / N * f((k + .5) / N);
    }
    if(k>0)
    {
      system_matrix.add(k, k-1, -N + 1./(6.*N));
      rhs(k) += .5 / N * f((k - .5) / N);
    }
  }
}


void solve()
{
  SolverControl  solver_control(1<<30, 1e-12);
  SolverRelaxation<Vector<double>>  solver(solver_control);

  // The SOR method with coefficient 1.0 is indeed Gauss-Seidel.
  PreconditionSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.0);

  solver.solve(system_matrix, solution, rhs, preconditioner);
  std::cout << "   " << solver_control.last_step()
            << " G-S iterations." << std::endl;
}


void output_error()
{
  double res = 0;
  std::ofstream fout("result.txt");
  for(int i = 0; i <= N; i++)
  {
    double err = solution[i] - u(1.*i/N);
    res += pow(err, 2.);
    fout << 1.*i/N << " " << solution[i] << " " << err << std::endl;
  }
  std::cout << "   discrete-L2 error: " << sqrt(res/N) << std::endl;
}


int main()
{
  CPUTimer timer_assemble;
    make_sparse_pattern();
    assemble_system();
  std::cout << "Assembled in " << timer_assemble() << "s." << std::endl;
  std::cout << "Solving..." << std::endl;

  CPUTimer timer_solve;
    solve();
  std::cout << "Solved in " << timer_solve() << "s." << std::endl;

  output_error();
  return 0;
}