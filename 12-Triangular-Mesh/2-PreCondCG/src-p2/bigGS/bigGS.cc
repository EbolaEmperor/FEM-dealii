#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_relaxation.h>

#include <fstream>

#include <chrono>

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

const int n_dof = 43653;

SparsityPattern sparsity_pattern;
SparseMatrix<double> system_matrix;
Vector<double> rhs;
Vector<double> solution;

const int Nj[] = {128, 512, 2048, 8192, 32768};
const int V = 1;

double u(const double &x)
{
  return cos(V * M_PI * x);
}


double f(const double &x)
{
  return (M_PI*M_PI*V*V + 1) * cos(V * M_PI * x);
}


int idx(const int &j, const int &k)
{
  if(j==0)      return k;
  else if(j==1) return 129 + k;
  else if(j==2) return 642 + k;
  else if(j==3) return 2691 + k;
  else if(j==4) return 10884 + k;
  else
  {
    std::cerr << "idx: out of range" << std::endl;
    exit(-1);
  }
  return -1;
}


void make_sparse_pattern()
{
  DynamicSparsityPattern dynamic_sparsity_pattern(n_dof, n_dof);
  for(int j = 0; j < 5; j++)
  {
    const int& N = Nj[j];
    for(int k = 0; k <= N; k++)
    {
      const int row = idx(j,k);
      dynamic_sparsity_pattern.add(row, row);
      if(k<N) dynamic_sparsity_pattern.add(row, row+1);
      if(k>0) dynamic_sparsity_pattern.add(row, row-1);
      int p = 1;
      for(int fj = j+1; fj < 5; fj++)
      {
        p *= 4;
        dynamic_sparsity_pattern.add(row, idx(fj,p*k));
        dynamic_sparsity_pattern.add(idx(fj,p*k), row);
        if(k<N)
        {
          dynamic_sparsity_pattern.add(row, idx(fj,p*(k+1)));
          dynamic_sparsity_pattern.add(idx(fj,p*(k+1)), row);
        }
        if(k>0)
        {
          dynamic_sparsity_pattern.add(row, idx(fj,p*(k-1)));
          dynamic_sparsity_pattern.add(idx(fj,p*(k-1)), row);
        }
        for(int v = -p; v <= p; v++)
        {
          if(p*k+v < 0 || p*k+v > Nj[fj]) continue;
          dynamic_sparsity_pattern.add(row, idx(fj,p*k+v));
          dynamic_sparsity_pattern.add(idx(fj,p*k+v), row);
        }
      }
    }
  }
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);
  rhs.reinit(n_dof);
  solution.reinit(n_dof);
}


double integerate(const double &lp,
                  const double &lval, 
                  const double &rval, 
                  const double &length, 
                  const int &N)
{
  double sum = 0;
  double unit_length = length / N;
  for(int i = 0; i < N; i++)
  {
    double x = (i+.5)/N;
    double v = (1.-x)*lval + x*rval;
    sum += unit_length * f(lp + (i+.5)*unit_length) * v;
  }
  return sum;
}


double base_func(const int &j, const int &k, const double &x)
{
  const int &N = Nj[j];
  if(x < -1e-15 || x > 1+1e-15) return 0;
  if(x < 1.*(k-1)/N || x > 1.*(k+1)/N) return 0;
  return 1. - fabs(x - 1.*k/N) * N;
}


double mass_product_each_inteval(const int &j, const int &k,
                                 const int &fj, const int &fk,
                                 const double &l, const double &r)
{
  static const double x1 = 0.5 - 0.5/sqrt(3);
  static const double x2 = 0.5 + 0.5/sqrt(3);
  return ( base_func(j, k, l + (r-l)*x1) * base_func(fj, fk, l + (r-l)*x1)
         + base_func(j, k, l + (r-l)*x2) * base_func(fj, fk, l + (r-l)*x2) ) * (r-l) / 2;
}


double mass_product(const int &j, const int &k,
                    const int &fj, const int &fk)
{
  return mass_product_each_inteval(j, k, fj, fk,
                                   (fk-1.)/Nj[fj], 1.*fk/Nj[fj])
       + mass_product_each_inteval(j, k, fj, fk,
                                   1.*fk/Nj[fj], (fk+1.)/Nj[fj]);
}


void assemble_system()
{
  for(int j = 0; j < 5; j++)
  {
    const int& N = Nj[j];
    for(int k = 0; k <= N; k++)
    {
      const int row = idx(j,k);
      int fg = (k==0 || k==N) ? 1 : 2;
      system_matrix.add(row, row, fg*N + fg/(3.*N));
      if(k<N)
      {
        system_matrix.add(row, row+1, -N + 1./(6.*N));
        rhs(row) += integerate(1.*k/N, 1., 0., 1./N, Nj[4]/N);
      }
      if(k>0)
      {
        system_matrix.add(row, row-1, -N + 1./(6.*N));
        rhs(row) += integerate(1.*(k-1)/N, 0., 1., 1./N, Nj[4]/N);
      }
      int p = 1;
      for(int fj = j+1; fj < 5; fj++)
      {
        p *= 4;
        system_matrix.add(row, idx(fj,p*k), fg*N);
        system_matrix.add(idx(fj,p*k), row, fg*N);
        if(k<N)
        {
          system_matrix.add(row, idx(fj,p*(k+1)), -N);
          system_matrix.add(idx(fj,p*(k+1)), row, -N);
        }
        if(k>0)
        {
          system_matrix.add(row, idx(fj,p*(k-1)), -N);
          system_matrix.add(idx(fj,p*(k-1)), row, -N);
        }
        for(int v = -p; v <= p; v++)
        {
          if(p*k+v < 0 || p*k+v > Nj[fj]) continue;
          system_matrix.add(row, idx(fj,p*k+v), mass_product(j,k,fj,p*k+v));
          system_matrix.add(idx(fj,p*k+v), row, mass_product(j,k,fj,p*k+v));
        }
      }
    }
  }

  std::ofstream fout("matrix.txt");
  system_matrix.print(fout);
}


void output_error()
{
  double res = 0;
  const int N = Nj[4];
  std::ofstream fout("result.txt");

  Vector<double> comb_solution;
  comb_solution.reinit(N+1);

  // Sum up all nodal basis to compute solution values.
  for(int j = 0; j < 5; j++)
  {
    const int intervals = N/Nj[j];
    for(int k = 0; k <= N; k++)
    {
      if(k % intervals == 0)
        comb_solution[k] += solution[idx(j,k/intervals)];
      else
      {
        double x = 1. * (k-k/intervals*intervals) / intervals;
        comb_solution[k] += solution[idx(j,k/intervals)] * (1.-x)
                          + solution[idx(j,k/intervals+1)] * x;
      }
    }
  }

  for(int i = 0; i <= N; i++)
  {
    double err = comb_solution[i] - u(1.*i/N);
    res += pow(err, 2.);
    fout << 1.*i/N << " " << comb_solution[i] << " " << err << std::endl;
  }
  std::cout << "   discrete-L2 error: " << sqrt(res/N) << std::endl;
}


void solve()
{ 
  SolverControl  solver_control(50000000, 1e-12);
  SolverRelaxation<Vector<double>>  solver(solver_control);

  // The SOR method with coefficient 1.0 is indeed Gauss-Seidel.
  PreconditionSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.0);

  solver.solve(system_matrix, solution, rhs, preconditioner);
  std::cout << "   " << solver_control.last_step()
            << " big G-S iterations." << std::endl;
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