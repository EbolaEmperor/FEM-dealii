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

// Our test velocity field is divergence-free.
#define DIVERGENCE_FREE

using namespace dealii;

const double diffusion_coefficient = 0.001;

template <int dim>
class VelocityField : public Function<dim>
{
public:
  virtual void vector_value(const Point<dim>  &p,
                        Vector<double> &values) const override;

#ifndef DIVERGENCE_FREE
  double divergence(const Point<dim>  &p) const;
#endif

};


template <int dim>
void VelocityField<dim>::vector_value(const Point<dim>  &p,
                        Vector<double> &values) const
{
  Assert(dim == 2, ExcNotImplemented());
  values.reinit(dim);
  values[0] = 0.1 * sin(M_PI*p[0]) * sin(M_PI*p[0]) * sin(2*M_PI*p[1]);
  values[1] = -0.1 * sin(2*M_PI*p[0]) * sin(M_PI*p[1]) * sin(M_PI*p[1]);
}


#ifndef DIVERGENCE_FREE
template <int dim>
double VelocityField<dim>::divergence(const Point<dim>  &p) const
{
  Assert(dim == 2, ExcNotImplemented());
  return 0;
}
#endif


template <int dim>
class InitialValues : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};


template <int dim>
double InitialValues<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  static const double cx = 0.5;
  static const double cy = 0.75;
  return exp( ((p[0]-cx)*(p[0]-cx) + (p[1]-cy)*(p[1]-cy)) / (0.01 / log(1e-16)) );
}

template <int dim>
class AdvectionDiffusionEquation{
public:
  AdvectionDiffusionEquation();
  void run();

private:
  void make_mesh();
  void refine_mesh(const unsigned& min_level, const unsigned& max_level);
  void setup_system();
  void solve_time_step(Vector<double>& solution);
  void output_result() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> laplace_matrix;

#ifndef DIVERGENCE_FREE
  SparseMatrix<double> divergence_velocity_matrix;
#endif
  
  SparseMatrix<double> velocity_gradient_matrix;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> prev_solution;
  Vector<double> system_rhs;

  double   time;
  double   time_step;
  unsigned timestep_number;
};


template <int dim>
AdvectionDiffusionEquation<dim>::AdvectionDiffusionEquation()
  : fe(1)
  , dof_handler(triangulation)
  , time_step(1. / 500)
{}


template <int dim>
void AdvectionDiffusionEquation<dim>::make_mesh(){
  GridGenerator::hyper_cube(triangulation);
}


template <int dim>
void AdvectionDiffusionEquation<dim>::refine_mesh(const unsigned &min_level, const unsigned &max_level)
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim-1>(fe.degree+1),
                                     {},                                 // Here for Neumann conditions.
                                     solution,
                                     estimated_error_per_cell);
  
  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.25,
                                                  0.2);
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
void AdvectionDiffusionEquation<dim>::output_result() const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("solution/solution-" + std::to_string(timestep_number) + ".vtu");
  data_out.write_vtu(output);
}


template <int dim>
void AdvectionDiffusionEquation<dim>::solve_time_step(Vector<double>& solution)
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
void AdvectionDiffusionEquation<dim>::setup_system()
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
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);

#ifndef DIVERGENCE_FREE
  divergence_velocity_matrix.reinit(sparsity_pattern);
#endif

  velocity_gradient_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  MatrixCreator::create_mass_matrix(dof_handler,
                                    QGauss<dim>(fe.degree+1),
                                    mass_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<dim>(fe.degree+1),
                                       laplace_matrix);
  laplace_matrix *= diffusion_coefficient;

  const unsigned dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix_1(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_2(dofs_per_cell, dofs_per_cell);

  FEValues<dim> fe_values(fe,
                          QGauss<dim>(fe.degree+1),
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
  VelocityField<dim> velocity;
  Vector<double> u;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix_1 = 0.;
      cell_matrix_2 = 0.;
      fe_values.reinit(cell);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {

#ifndef DIVERGENCE_FREE
          double divu = velocity.divergence( fe_values.quadrature_point(q_index) );
#endif

          velocity.vector_value( fe_values.quadrature_point(q_index), u );

          for (const unsigned int i : fe_values.dof_indices())
            {
              for (const unsigned int j : fe_values.dof_indices())
              {

#ifndef DIVERGENCE_FREE
                cell_matrix_1(j, i) +=
                  (divu                             *  // div(u)
                   fe_values.shape_value(i, q_index) * // phi_i(x_q)
                   fe_values.shape_value(j, q_index) * // phi_j(x_q)
                   fe_values.JxW(q_index));            // dx
#endif

                auto gradj = fe_values.shape_grad(i, q_index);
                cell_matrix_2(j, i) +=
                  ((u[0] * gradj[0] + u[1] * gradj[1])  *  // u(x_q) dot grad_phi_j(x_q)
                    fe_values.shape_value(j, q_index) * // phi_j(x_q)
                    fe_values.JxW(q_index));            // dx
              }
            }
      }
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
        {

#ifndef DIVERGENCE_FREE
          divergence_velocity_matrix.add(local_dof_indices[i],
                                         local_dof_indices[j],
                                         cell_matrix_1(i, j));
#endif
          
          velocity_gradient_matrix.add(local_dof_indices[i],
                                       local_dof_indices[j],
                                       cell_matrix_2(i, j));
        }
    }
  
  solution.reinit(dof_handler.n_dofs());
  prev_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


template <int dim>
void AdvectionDiffusionEquation<dim>::run(){
  const unsigned initial_global_refinement = 4;
  const unsigned n_adaptive_pre_refinement_steps = 5;
  Vector<double> tmp;
  Vector<double> middle_solution;
  double end_time = 10.0;
  unsigned pre_refinement_step = 0;

  make_mesh();
  triangulation.refine_global(initial_global_refinement);
  setup_system();

start_time_iteration:
  time            = 0.0;
  timestep_number = 0;

  tmp.reinit(solution.size());
  middle_solution.reinit(solution.size());
  VectorTools::interpolate(dof_handler,
                           InitialValues<dim>(),
                           prev_solution);
  solution = prev_solution;
  output_result();

  while(time <= end_time){
    time += time_step;
    timestep_number++;
    std::cout << "Time step " << timestep_number << " at t=" << time << std::endl;

    //------------------------------first stage-------------------------------------

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(0.5*time_step, laplace_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution);

#ifndef DIVERGENCE_FREE
    divergence_velocity_matrix.vmult(tmp, prev_solution);
    system_rhs.add(-time_step, tmp);
#endif

    velocity_gradient_matrix.vmult(tmp, prev_solution);
    system_rhs.add(-time_step, tmp);
    laplace_matrix.vmult(tmp, prev_solution);
    system_rhs.add(-0.5*time_step, tmp);

    // constriant hanging verteces
    constraints.condense(system_matrix, system_rhs);
    
    solve_time_step(middle_solution);


    //------------------------------second stage-------------------------------------

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);

    // setup right side term
    middle_solution += prev_solution;
    mass_matrix.vmult(system_rhs, prev_solution);

#ifndef DIVERGENCE_FREE
    divergence_velocity_matrix.vmult(tmp, middle_solution);
    system_rhs.add(-0.5*time_step, tmp);
#endif

    velocity_gradient_matrix.vmult(tmp, middle_solution);
    system_rhs.add(-0.5*time_step, tmp);
    laplace_matrix.vmult(tmp, middle_solution);
    system_rhs.add(-0.5*time_step, tmp);

    constraints.condense(system_matrix, system_rhs);
    solve_time_step(solution);
    if(timestep_number % 10 == 0) output_result();

    // Refine mesh 1 times per 5 time-step. At the begining, refine 5 times.
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
    else if ((timestep_number > 0) && (timestep_number % 10 == 0))
      {
        refine_mesh(initial_global_refinement,
                    initial_global_refinement +
                      n_adaptive_pre_refinement_steps);
        tmp.reinit(solution.size());
        middle_solution.reinit(solution.size());
      }

    prev_solution = solution;
  }
}


int main(){
  AdvectionDiffusionEquation<2> advection_diffusion_equation;
  advection_diffusion_equation.run();
  return 0;
}
