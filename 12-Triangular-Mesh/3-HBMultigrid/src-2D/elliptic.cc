#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/dofs/dof_renumbering.h>

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
 
#include <deal.II/numerics/data_out.h>

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

int fineast_level, level_num;

struct Vertex
{
  Point<2> pos;
  unsigned idx;

  int dcmp(const double &a, const double &b) const
  {
    if(fabs(a-b) < 1e-12) return 0;
    else if(a < b) return -1;
    else return 1;
  }

  bool operator < (const Vertex &rhs) const
  {
    return dcmp(pos[0], rhs.pos[0])==-1
        || ( dcmp(pos[0], rhs.pos[0])==0
          && dcmp(pos[1], rhs.pos[1])==-1 );
  }
};


class HiercharchicalPreconditioner : public Subscriptor
{
public:
  HiercharchicalPreconditioner(const int& fine_level,
                               const int& fine_n_dofs,
                               const SparseMatrix<double>& fine_system_matrix);
  void initialize();
  void vmult(Vector<double> &dst, const Vector<double> &src) const;

private:
  int level, pow2level;

  Triangulation<2> triangulation;
  FE_SimplexP<2>   fe;
  DoFHandler<2>    dof_handler;
  std::map<types::global_dof_index, Point<2>> dof_location_map;

  AffineConstraints<double> constraints;
 
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
 
  Vector<double> diag_D;
  HiercharchicalPreconditioner *preconditioner;

  unsigned corse2fine(const unsigned &i) const{
    return 2*(1+2*pow2level)*(i/(1+pow2level)) + 2*(i%(1+pow2level));
  }
};


HiercharchicalPreconditioner::
  HiercharchicalPreconditioner(const int& fine_level,
                               const int& fine_n_dofs,
                               const SparseMatrix<double>& fine_system_matrix):
  level(fine_level-1),
  pow2level(1<<level),
  fe(1),
  dof_handler(triangulation)
{
  diag_D.reinit(fine_n_dofs);
  for(int i = 0; i < fine_n_dofs; i++)
    diag_D[i] = fine_system_matrix(i, i);
}


void HiercharchicalPreconditioner::initialize()
{
  GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, 1);
  triangulation.refine_global(level);
  dof_handler.distribute_dofs(fe);
  dof_location_map = DoFTools::map_dofs_to_support_points(MappingFE<2>(fe), dof_handler);
  
  // Renumber the fucking DoFs !!!
  std::vector<Vertex> fucking_dofs(dof_handler.n_dofs());
  for(unsigned i = 0; i < fucking_dofs.size(); i++)
  {
    fucking_dofs[i].pos = dof_location_map[i];
    fucking_dofs[i].idx = i;
  }
  std::sort(fucking_dofs.begin(), fucking_dofs.end());
  std::vector< types::global_dof_index > new_numbers(dof_handler.n_dofs());
  for(unsigned i = 0; i < fucking_dofs.size(); i++)
    new_numbers[fucking_dofs[i].idx] = i;
  dof_handler.renumber_dofs(new_numbers);
  dof_location_map = DoFTools::map_dofs_to_support_points(MappingFE<2>(fe), dof_handler);

  // Add the homogeneous Dirichlet constraint
  constraints.clear();
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                  dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                      QGaussSimplex<2>(fe.degree),
                                      system_matrix);
  constraints.condense(system_matrix);
  
  for(unsigned i = 0; i < dof_handler.n_dofs(); i++)
    diag_D[corse2fine(i)] = 0;
  
  if(level >= 4 && level > fineast_level - level_num + 1){
    preconditioner = new HiercharchicalPreconditioner(level, dof_handler.n_dofs(), system_matrix);
    preconditioner->initialize();
  } else {
    preconditioner = nullptr;
  }
}


void HiercharchicalPreconditioner::
  vmult(Vector<double> &dst, const Vector<double> &src) const
{
  Vector<double> system_rhs;
  system_rhs.reinit(dof_handler.n_dofs());

  Vector<double> solution;
  solution.reinit(dof_handler.n_dofs());
  
  // full-weighting
  const unsigned pw = pow2level<<1;
  auto it = dof_location_map.begin();
  for(unsigned i = 0; i < system_rhs.size(); i++, it++){
    int cf = corse2fine(i);
    const auto& pos = it->second;
    system_rhs[i] =  0.25 * src[cf];
    if(pos[1] > 1e-8) system_rhs[i] += 0.125 * src[cf-1];
    if(pos[1] < 1 - 1e-8) system_rhs[i] += 0.125 * src[cf+1];
    if(pos[0] > 1e-8) system_rhs[i] += 0.125 * src[cf-pw-1];
    if(pos[0] < 1 - 1e-8) system_rhs[i] += 0.125 * src[cf+pw+1];
    if(pos[0] > 1e-8 && pos[1] < 1 - 1e-8) system_rhs[i] += 0.125 * src[cf-pw];
    if(pos[1] > 1e-8 && pos[0] < 1 - 1e-8) system_rhs[i] += 0.125 * src[cf+pw];
  }
  
  SolverControl            solver_control(2000, .05 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);

  constraints.condense(system_rhs);
  if(preconditioner == nullptr)
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  else
    solver.solve(system_matrix, solution, system_rhs, *preconditioner);
  constraints.distribute(solution);

  // std::cout << "     " << solver_control.last_step()
  //           << " inner CG iterations." << std::endl;

  dst.reinit(src.size());
  for(unsigned i = 0; i < dst.size(); i++)
    if(diag_D[i]) dst[i] = src[i] / diag_D[i];
  
  // linear interpolation
  it = dof_location_map.begin();
  for(unsigned i = 0; i < system_rhs.size(); i++, it++){
    int cf = corse2fine(i);
    const auto& pos = it->second;
    dst[cf] += solution[i];
    if(pos[1] > 1e-8) dst[cf-1] += .5 * solution[i];
    if(pos[1] < 1 - 1e-8) dst[cf+1] += .5 * solution[i];
    if(pos[0] > 1e-8) dst[cf-pw-1] += .5 * solution[i];
    if(pos[0] < 1 - 1e-8) dst[cf+pw+1] += .5 * solution[i];
    if(pos[0] > 1e-8 && pos[1] < 1 - 1e-8) dst[cf-pw] += .5 * solution[i];
    if(pos[1] > 1e-8 && pos[0] < 1 - 1e-8) dst[cf+pw] += .5 * solution[i];
  }
}


class RHS : public Function<2>{
public:
  double value(const Point<2> & p, const unsigned int component = 0) const override{
    (void)component;
    return 2. * M_PI * M_PI * sin(M_PI*p[0]) * sin(M_PI*p[1]);
  };
};


class Solution : public Function<2>{
public:
  double value(const Point<2> & p, const unsigned int component = 0) const override{
    (void)component;
    return sin(M_PI*p[0]) * sin(M_PI*p[1]);
  };

  virtual Tensor<1, 2>
  gradient(const Point<2> & p, const unsigned int component = 0) const override{
    (void)component;
    Tensor<1, 2> grad;
    grad[0] = M_PI * cos(M_PI*p[0]) * sin(M_PI*p[1]);
    grad[1] = M_PI * sin(M_PI*p[0]) * cos(M_PI*p[1]);
    return grad;
  };
};


class Elliptic{
public:
  Elliptic(const int &level);
  void run();

private:
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  int level;

  Triangulation<2> triangulation;
  FE_SimplexP<2>   fe;
  DoFHandler<2>    dof_handler;
  std::map<types::global_dof_index, Point<2>> dof_location_map;

  AffineConstraints<double> constraints;
 
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
 
  Vector<double> solution;
  Vector<double> error;
  Vector<double> system_rhs;

  CPUTimer timer;
};


Elliptic::Elliptic(const int &level): 
  level(level),
  fe(1), 
  dof_handler(triangulation)
{}


void Elliptic::setup_system(){
  GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, 1);
  triangulation.refine_global(level);
  dof_handler.distribute_dofs(fe);
  dof_location_map = DoFTools::map_dofs_to_support_points(MappingFE<2>(fe), dof_handler);
  
  // Renumber the fucking DoFs !!!
  std::vector<Vertex> fucking_dofs(dof_handler.n_dofs());
  for(unsigned i = 0; i < fucking_dofs.size(); i++)
  {
    fucking_dofs[i].pos = dof_location_map[i];
    fucking_dofs[i].idx = i;
  }
  std::sort(fucking_dofs.begin(), fucking_dofs.end());
  std::vector< types::global_dof_index > new_numbers(dof_handler.n_dofs());
  for(unsigned i = 0; i < fucking_dofs.size(); i++)
    new_numbers[fucking_dofs[i].idx] = i;
  dof_handler.renumber_dofs(new_numbers);
  dof_location_map = DoFTools::map_dofs_to_support_points(MappingFE<2>(fe), dof_handler);

  // Add the homogeneous Dirichlet constraint
  constraints.clear();
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                  dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  error.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  MatrixCreator::create_laplace_matrix(dof_handler,
                                      QGaussSimplex<2>(fe.degree),
                                      system_matrix);
  VectorTools::create_right_hand_side(dof_handler,
                                      QGaussSimplex<2>(fe.degree),
                                      RHS(),
                                      system_rhs);
  constraints.condense(system_matrix, system_rhs);
}


void Elliptic::solve()
{
  SolverControl            solver_control(20000, 1e-12 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);
  
  if(level_num > 1)
  { // Solve with multi-grid preconditioned CG
    HiercharchicalPreconditioner preconditioner(level,
                                                dof_handler.n_dofs(),
                                                system_matrix);
    preconditioner.initialize();
    std::cout << "Pre-processed in " << timer() << "s." << std::endl;
    timer.reset();
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
  }
  else
  { // Solve with naive CG
    std::cout << "Pre-processed in " << timer() << "s." << std::endl;
    timer.reset();
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  }
  
  constraints.distribute(solution);
  std::cout << "   " << solver_control.last_step()
            << " CG iterations." << std::endl;
  std::cout << "Solved in " << timer() << "s." << std::endl;
  timer.reset();

  Solution u;
  for(unsigned i = 0; i < solution.size(); i++){
    error[i] = solution[i] - u.value(dof_location_map[i]);
  }
}


void Elliptic::output_results() const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.add_data_vector(error, "error");
  data_out.build_patches();
  std::ofstream output("solution.vtk");
  data_out.write_vtk(output);

  Vector<double> difference_per_cell(triangulation.n_active_cells());

  VectorTools::integrate_difference(MappingFE<2>(fe),
                                    dof_handler,
                                    solution,
                                    Solution(),
                                    difference_per_cell,
                                    QGaussSimplex<2>(fe.degree + 1),
                                    VectorTools::L2_norm);
  const double L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
  std::cout << "   L2-norm error: " << L2_error << std::endl;

  difference_per_cell.reinit(triangulation.n_active_cells());
  const QTrapezoid<1>  q_trapez;
  const QIterated<2> q_iterated(q_trapez, fe.degree*2+1);
  VectorTools::integrate_difference(MappingFE<2>(fe),
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
  std::cout << "   Linfty-norm error: " << Linf_error << std::endl;

  difference_per_cell.reinit(triangulation.n_active_cells());
  VectorTools::integrate_difference(MappingFE<2>(fe),
                                    dof_handler,
                                    solution,
                                    Solution(),
                                    difference_per_cell,
                                    QGaussSimplex<2>(fe.degree + 1),
                                    VectorTools::H1_norm);
  const double H1_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::H1_norm);
  std::cout << "   H1-norm error: " << H1_error << std::endl;
  std::cout << "Post-processed in " << timer() << "s." << std::endl;
}


void Elliptic::run()
{
  setup_system();
  solve();
  output_results();
}

 
int main(int argc, char* argv[])
{
  if(argc < 2){
    std::cerr << "Param error!" << std::endl;
    return -1;
  }
  fineast_level = std::stoi(argv[1]);
  level_num = (argc >= 3) ? std::stoi(argv[2]) : 1;
  Elliptic solver(fineast_level);
  solver.run();
  return 0;
}