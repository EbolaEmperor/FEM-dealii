#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide()
    : Function<dim>()
    , period(0.2)
  {}

  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;

private:
  const double period;
};


template <int dim>
double RightHandSide<dim>::value(const Point<dim>  &p,
                                  const unsigned int component) const
{
  (void)component;
  AssertIndexRange(component, 1);
  Assert(dim == 2, ExcNotImplemented());

  const double scale = 3;
  const double time = this->get_time();
  const double point_within_period =
    (time / period - std::floor(time / period));

  if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
    {
      if ((p[0] > 0) && (p[1] > 0))
        return scale;
      else
        return 0;
    }
  else if ((point_within_period >= 0.25) && (point_within_period <= 0.45))
    {
      if ((p[0] > 0) && (p[1] < 0))
        return scale;
      else
        return 0;
    }
  else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
    {
      if ((p[0] < 0) && (p[1] < 0))
        return scale;
      else
        return 0;
    }
  else if ((point_within_period >= 0.75) && (point_within_period <= 0.95))
    {
      if ((p[0] < 0) && (p[1] > 0))
        return scale;
      else
        return 0;
    }
  else
    return 0;
}


template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};


template <int dim>
double BoundaryValues<dim>::value(const Point<dim> & /*p*/,
                                  const unsigned int component) const
{
  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return 0;
}

template <int dim>
class HeatEquation{
public:
  HeatEquation();
  void run();

private:
  void make_mesh();
  void refine_mesh(const unsigned& min_level, const unsigned& max_level);
  void setup_system();
  void solve_time_step();
  void output_result() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> laplace_matrix;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> prev_solution;
  Vector<double> system_rhs;

  double   time;
  double   time_step;
  unsigned timestep_number;

  const double theta;
};


template <int dim>
HeatEquation<dim>::HeatEquation()
  : fe(1)
  , dof_handler(triangulation)
  , time_step(1. / 500)
  , theta(0.5)
{}


template <int dim>
void HeatEquation<dim>::make_mesh(){
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("circle-grid.inp");

  grid_in.read_ucd(input_file);
  const SphericalManifold<dim> boundary;
  triangulation.set_all_manifold_ids_on_boundary(0);

  for(Triangulation<2>::cell_iterator cell = triangulation.begin(); cell != triangulation.end(); cell++){
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f){
      bool is_inner_face = true;
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v){
        Point<dim> &vertex = cell->face(f)->vertex(v);
        if (std::abs(vertex.norm() - 1.0) > 0.1)
        {
          is_inner_face = false;
          break;
        }
      }
      if (is_inner_face)
        cell->face(f)->set_manifold_id(1);
    }
  }
  triangulation.set_manifold(1, boundary);
}


template <int dim>
void HeatEquation<dim>::refine_mesh(const unsigned &min_level, const unsigned &max_level)
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim-1>(fe.degree+1),
                                     {},                                 // Here for Neumann conditions.
                                     solution,
                                     estimated_error_per_cell);
  
  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.6,                       // Refine 60% cells
                                                  0.4);                      // Coarsen 40% cells
  if(triangulation.n_levels() > max_level)
    for(const auto &cell : triangulation.active_cell_iterators_on_level(max_level))
      cell->clear_refine_flag();
  for(const auto &cell : triangulation.active_cell_iterators_on_level(min_level))
    cell->clear_coarsen_flag();
  
  SolutionTransfer<dim> solution_trans(dof_handler);
  Vector<double> previous_solution;
  previous_solution = solution;
  triangulation.prepare_coarsening_and_refinement();
  solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
  triangulation.execute_coarsening_and_refinement();
  setup_system();
  solution_trans.interpolate(previous_solution, solution);
  constraints.distribute(solution);
}


template <int dim>
void HeatEquation<dim>::output_result() const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("solution-" + std::to_string(timestep_number) + ".vtu");
  data_out.write_vtu(output);
}


template <int dim>
void HeatEquation<dim>::solve_time_step()
{
  SolverControl            solver_control(1000, 
                                          1e-8 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.0);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  std::cout << "   " << solver_control.last_step()
            << " CG iterations." << std::endl;
  constraints.distribute(solution);
}


template <int dim>
void HeatEquation<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << std::endl
            << "===========================================" << std::endl
            << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           BoundaryValues<dim>(),
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  MatrixCreator::create_mass_matrix(dof_handler,
                                    QGauss<dim>(fe.degree+1),
                                    mass_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<dim>(fe.degree+1),
                                       laplace_matrix);
  
  solution.reinit(dof_handler.n_dofs());
  prev_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


template <int dim>
void HeatEquation<dim>::run(){
  const unsigned initial_global_refinement = 3;
  const unsigned n_adaptive_pre_refinement_steps = 4;
  Vector<double> tmp;
  Vector<double> forcing_terms;
  double end_time = 1.0;
  unsigned pre_refinement_step = 0;

  make_mesh();
  triangulation.refine_global(initial_global_refinement);
  setup_system();

start_time_iteration:
  time            = 0.0;
  timestep_number = 0;

  tmp.reinit(solution.size());
  forcing_terms.reinit(solution.size());
  VectorTools::interpolate(dof_handler,
                           Functions::ZeroFunction<dim>(),
                           prev_solution);
  solution = prev_solution;
  output_result();

  while(time <= end_time){
    time += time_step;
    timestep_number++;
    std::cout << "Time step " << timestep_number << " at t=" << time << std::endl;

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(time_step*theta, laplace_matrix);

    // setup system_rhs
    mass_matrix.vmult(system_rhs, prev_solution);
    laplace_matrix.vmult(tmp, prev_solution);
    system_rhs.add(-time_step*(1-theta), tmp);

    RightHandSide<dim> rhs_func;
    rhs_func.set_time(time);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree+1),
                                        rhs_func,
                                        forcing_terms);
    forcing_terms *= time_step * theta;
    rhs_func.set_time(time-time_step);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree+1),
                                        rhs_func,
                                        tmp);
    forcing_terms.add(time_step*(1-theta), tmp);
    system_rhs += forcing_terms;

    // constriant hanging verteces
    constraints.condense(system_matrix, system_rhs);
    
    solve_time_step();
    output_result();

    // Refine mesh 1 times per 5 time-step. At the begining, refine 4 times.
    if ((timestep_number == 1) &&
        (pre_refinement_step < n_adaptive_pre_refinement_steps))
      {
        refine_mesh(initial_global_refinement,
                    initial_global_refinement +
                      n_adaptive_pre_refinement_steps);
        ++pre_refinement_step;
        std::cout << std::endl;
        goto start_time_iteration;
      }
    else if ((timestep_number > 0) && (timestep_number % 5 == 0))
      {
        refine_mesh(initial_global_refinement,
                    initial_global_refinement +
                      n_adaptive_pre_refinement_steps);
        tmp.reinit(solution.size());
        forcing_terms.reinit(solution.size());
      }

    prev_solution = solution;
  }
}


int main(){
  HeatEquation<2> heat_equation;
  heat_equation.run();
  return 0;
}
