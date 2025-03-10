#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
 
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
 
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

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

class Solution : public Function<1>{
public:
  double value(const Point<1> & p, const unsigned int component = 0) const override{
    (void)component;
    if(p[0]<=0.) return 0.;
    return p[0]*log(p[0]);
  }

  virtual Tensor<1, 1>
  gradient(const Point<1> & p, const unsigned int component = 0) const override{
    (void)component;
    Tensor<1, 1> grad;
    if(p[0]<=0.) grad[0] = 1.;
    else grad[0] = log(p[0]) + 1.;
    return grad;
  }
};

double f(const double &x){
  return -1./x;
}

class Elliptic{
public:
  Elliptic(const int &level, const bool &uniform = true);
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  int level;
  Triangulation<1> triangulation;
  FE_Q<1>          fe;
  DoFHandler<1>    dof_handler;
  std::map<types::global_dof_index, Point<1>> dof_location_map;

  AffineConstraints<double> constraints;
 
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> mesh_points;
  Vector<double> solution;
  Vector<double> error;
  Vector<double> system_rhs;

  bool uniform;
};


Elliptic::Elliptic(const int &level, const bool& uniform): 
  level(level), 
  fe(1), 
  dof_handler(triangulation),
  uniform(uniform)
{}


void Elliptic::make_grid(){
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(level);
  if(uniform) return;

  for (Triangulation<1>::cell_iterator cell = triangulation.begin_active();
       cell != triangulation.end();
       ++cell)
    {
      Point<1> &vertex = cell->vertex(0);
      vertex[0] = vertex[0] * vertex[0];
    }
}


void Elliptic::setup_system(){
  dof_handler.distribute_dofs(fe);
  std::cout << "   " << dof_handler.n_dofs() << " DoF numbers." << std::endl;

  constraints.clear();
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<1>(),
                                           constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           Functions::ZeroFunction<1>(),
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                  dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);

  system_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<1>(fe.degree+1),
                                       system_matrix);
 
  solution.reinit(dof_handler.n_dofs());
  error.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  mesh_points.reinit(dof_handler.n_dofs());
}

 
void Elliptic::assemble_system()
{
  QGauss<1> quadrature_formula(fe.degree);
  FEValues<1> fe_values(fe,
                        quadrature_formula,
                        update_values | update_quadrature_points | update_JxW_values);
 
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
 
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_rhs    = 0;
      const auto &quadrature_points =  fe_values.get_quadrature_points();
      for (const unsigned int i : fe_values.dof_indices())
        cell_rhs(i) += fe_values.shape_value(i, 0)
                        * f(quadrature_points[0][0])
                        * fe_values.JxW(0);
      cell->get_dof_indices(local_dof_indices);
 
      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
}


void Elliptic::solve()
{
  SolverControl            solver_control(500000, 1e-6 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.999999999999);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  std::cout << "   " << solver_control.last_step()
            << " CG iterations." << std::endl;

  Solution u;
  for(unsigned i = 0; i < solution.size(); i++){
    error[i] = solution[i] - u.value(dof_location_map[i]);
    mesh_points[i] = pow(1.*i/(solution.size()-1), uniform ? 1. : 2.);
  }
}


void Elliptic::output_results() const
{
  std::ofstream fout("result.txt");
  for(unsigned i = 0; i < solution.size(); i++)
    fout << mesh_points[i] << " " 
         << solution[i] << " " 
         << error[i] << std::endl;

  Vector<double> difference_per_cell(triangulation.n_active_cells());
  difference_per_cell.reinit(triangulation.n_active_cells());
  VectorTools::integrate_difference(MappingFE<1>(fe),
                                    dof_handler,
                                    solution,
                                    Solution(),
                                    difference_per_cell,
                                    QGauss<1>(fe.degree + 1),
                                    VectorTools::L2_norm);
  const double L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
  std::cout << "L2-norm error: " << L2_error << std::endl;

  difference_per_cell.reinit(triangulation.n_active_cells());
  const QTrapezoid<1>  q_trapez;
  const QIterated<1> q_iterated(q_trapez, fe.degree*2+1);
  VectorTools::integrate_difference(MappingFE<1>(fe),
                                    dof_handler,
                                    solution,
                                    Solution(),
                                    difference_per_cell,
                                    q_iterated,
                                    VectorTools::Linfty_norm);
  const double Linf_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::Linfty_norm);
  std::cout << "Linfty-norm error: " << Linf_error << std::endl;

  difference_per_cell.reinit(triangulation.n_active_cells());
  VectorTools::integrate_difference(MappingFE<1>(fe),
                                    dof_handler,
                                    solution,
                                    Solution(),
                                    difference_per_cell,
                                    QGauss<1>(fe.degree + 1),
                                    VectorTools::H1_norm);
  const double H1_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::H1_norm);
  std::cout << "H1-norm error: " << H1_error << std::endl;
}


void Elliptic::run()
{
  CPUTimer timer_mg;
  make_grid();
  std::cout << "Making-grid time: " << timer_mg() << std::endl;

  CPUTimer timer_assemble;
  setup_system();
  assemble_system();
  std::cout << "Assembling time: " << timer_assemble() << std::endl;

  CPUTimer timer_solve;
  constraints.condense(system_matrix, system_rhs);
  solve();
  constraints.distribute(solution);
  std::cout << "Solving time: " << timer_solve() << std::endl << std::endl;

  CPUTimer timer_output;
  output_results();
  std::cout << "Computing error and outputing time: " << timer_output() << std::endl;
}


int main(int argc, char* argv[])
{
  if(argc < 2){
    std::cerr << "Param error!" << std::endl;
    return -1;
  }
  bool uniform = true;
  if(argc==3 && argv[2][0]=='u') uniform = false;
  Elliptic solver(std::stoi(argv[1]), uniform);
  solver.run();
  return 0;
}