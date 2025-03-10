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
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

const double diffusion_coefficient = 0.01;

//----------------------------Custom Settings------------------------------


template <int dim>
class TrueSolution1 : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};


template <int dim>
class TrueSolution2 : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};


template <int dim>
double TrueSolution1<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return cos(M_PI * this->get_time()) * sin(M_PI*p[0]) * sin(M_PI*p[0]) * sin(2*M_PI*p[1]);
}


template <int dim>
double TrueSolution2<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return - cos(M_PI * this->get_time()) * sin(2*M_PI*p[0]) * sin(M_PI*p[1]) * sin(M_PI*p[1]);
}


template <int dim>
class ForcingTerm1 : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};


template <int dim>
class ForcingTerm2 : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};


template <int dim>
double ForcingTerm1<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return - M_PI * sin(M_PI* this->get_time()) * pow(sin(M_PI*p[0]), 2) * sin(2*M_PI*p[1])

#ifndef NO_CONVECTION
         + M_PI * pow(cos(M_PI* this->get_time()), 2) * pow(sin(M_PI*p[0]), 2) * sin(2*M_PI*p[0]) * pow(sin(2*M_PI*p[1]), 2)
         - 2*M_PI * pow(cos(M_PI* this->get_time()), 2) * sin(2*M_PI*p[0]) * pow(sin(M_PI*p[0]), 2) * pow(sin(M_PI*p[1]), 2) * cos(2*M_PI*p[1])
#endif

         - 2*diffusion_coefficient*M_PI*M_PI * cos(M_PI* this->get_time()) * cos(2*M_PI*p[0]) * sin(2*M_PI*p[1])
         + 4*diffusion_coefficient*M_PI*M_PI * cos(M_PI* this->get_time()) * pow(sin(M_PI*p[0]), 2) * sin(2*M_PI*p[1]);
}


template <int dim>
double ForcingTerm2<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return   M_PI * sin(M_PI* this->get_time()) * sin(2*M_PI*p[0]) * pow(sin(M_PI*p[1]), 2)

#ifndef NO_CONVECTION
         - 2*M_PI * pow(cos(M_PI* this->get_time()), 2) * pow(sin(M_PI*p[0]), 2) * cos(2*M_PI*p[0]) * pow(sin(M_PI*p[1]), 2) * sin(2*M_PI*p[1])
         + M_PI * pow(cos(M_PI* this->get_time()), 2) * pow(sin(2*M_PI*p[0]), 2) * pow(sin(M_PI*p[1]), 2) * sin(2*M_PI*p[1])
#endif

         + 2*diffusion_coefficient*M_PI*M_PI * cos(M_PI* this->get_time()) * sin(2*M_PI*p[0]) * cos(2*M_PI*p[1])
         - 4*diffusion_coefficient*M_PI*M_PI * cos(M_PI* this->get_time()) * sin(2*M_PI*p[0]) * pow(sin(M_PI*p[1]), 2);
}


//---------------------------------Solver Class-----------------------------------


template <int dim>
class ConvectionDiffusionEquation{
public:
  ConvectionDiffusionEquation(const int &, const double &);
  void run();

private:
  void make_mesh();
  void refine_mesh(const unsigned& min_level, const unsigned& max_level);
  void setup_system();
  void setup_convection(const Vector<double>& u1, const Vector<double>& u2);
  void solve_time_step(Vector<double>& solution);
  void output_result() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> laplace_matrix;
  SparseMatrix<double> system_matrix;

  Vector<double> convection_u1;
  Vector<double> convection_u2;

  Vector<double> solution_u1;
  Vector<double> solution_u2;
  Vector<double> prev_solution_u1;
  Vector<double> prev_solution_u2;
  Vector<double> system_rhs;

  int      level;
  double   end_time;
  double   time;
  double   time_step;
  unsigned timestep_number;
};


//---------------------------------Solver Implemention-----------------------------------


template <int dim>
ConvectionDiffusionEquation<dim>::ConvectionDiffusionEquation
  (const int &N, const double &T)
  : fe(1)
  , dof_handler(triangulation)
  , level(N)
  , end_time(T)
  , time_step(1. / (500 * (1<<N)))
{}


template <int dim>
void ConvectionDiffusionEquation<dim>::make_mesh(){
  GridGenerator::hyper_cube(triangulation);
}


template <int dim>
void ConvectionDiffusionEquation<dim>::refine_mesh(const unsigned &min_level, const unsigned &max_level)
{
  Vector<float> estimated_error_u1_per_cell(triangulation.n_active_cells());
  Vector<float> estimated_error_u2_per_cell(triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim-1>(fe.degree+1),
                                     {},                                 // Here for Neumann conditions.
                                     solution_u1,
                                     estimated_error_u1_per_cell);
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim-1>(fe.degree+1),
                                     {},                                 // Here for Neumann conditions.
                                     solution_u2,
                                     estimated_error_u2_per_cell);
  // Combine the results to compute the error of u=(u1,u2)
  for(unsigned i = 0; i < estimated_error_u1_per_cell.size(); i++)
  {
    estimated_error_u1_per_cell[i] = estimated_error_u1_per_cell[i] * estimated_error_u1_per_cell[i]
                                   + estimated_error_u2_per_cell[i] * estimated_error_u2_per_cell[i];
  }
  
  GridRefinement::refine_and_coarsen_optimize(triangulation,
                                              estimated_error_u1_per_cell);
  if(triangulation.n_levels() > max_level)
    for(const auto &cell : triangulation.active_cell_iterators_on_level(max_level))
      cell->clear_refine_flag();
  for(const auto &cell : triangulation.active_cell_iterators_on_level(min_level))
    cell->clear_coarsen_flag();
  
  SolutionTransfer<dim> solution_trans_u1(dof_handler);
  SolutionTransfer<dim> solution_trans_u2(dof_handler);
  Vector<double> previous_solution_u1;
  Vector<double> previous_solution_u2;
  previous_solution_u1 = solution_u1;
  previous_solution_u2 = solution_u2;
  triangulation.prepare_coarsening_and_refinement();
  solution_trans_u1.prepare_for_coarsening_and_refinement(previous_solution_u1);
  solution_trans_u2.prepare_for_coarsening_and_refinement(previous_solution_u2);
  triangulation.execute_coarsening_and_refinement();
  setup_system();
  solution_trans_u1.interpolate(previous_solution_u1, solution_u1);
  solution_trans_u2.interpolate(previous_solution_u2, solution_u2);
  constraints.distribute(solution_u1);
  constraints.distribute(solution_u2);
}


template <int dim>
void ConvectionDiffusionEquation<dim>::output_result() const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_u1, "solution_u1");
  data_out.add_data_vector(solution_u2, "solution_u2");

  Vector<double> difference_per_cell(triangulation.n_active_cells());
  TrueSolution1<dim> u1;
  u1.set_time(time);
  VectorTools::integrate_difference(MappingFE<2>(fe),
                                    dof_handler,
                                    solution_u1,
                                    u1,
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree + 1),
                                    VectorTools::L2_norm);
  double L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
  data_out.add_data_vector(difference_per_cell, "error_u1");
  std::cout << "u1 L2-norm error: " << L2_error << std::endl;

  difference_per_cell.reinit(triangulation.n_active_cells());
  TrueSolution2<dim> u2;
  u2.set_time(time);
  VectorTools::integrate_difference(MappingFE<2>(fe),
                                    dof_handler,
                                    solution_u2,
                                    u2,
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree + 1),
                                    VectorTools::L2_norm);
  L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
  data_out.add_data_vector(difference_per_cell, "error_u2");
  std::cout << "u2 L2-norm error: " << L2_error << std::endl;

  data_out.build_patches();
  std::ofstream output("solution/solution-" + std::to_string(timestep_number) + ".vtu");
  data_out.write_vtu(output);
}


template <int dim>
void ConvectionDiffusionEquation<dim>::solve_time_step(Vector<double>& solution)
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
void ConvectionDiffusionEquation<dim>::setup_system()
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
  system_matrix.reinit(sparsity_pattern);

  MatrixCreator::create_mass_matrix(dof_handler,
                                    QGauss<dim>(fe.degree+1),
                                    mass_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<dim>(fe.degree+1),
                                       laplace_matrix);
  laplace_matrix *= diffusion_coefficient;
  
  solution_u1.reinit(dof_handler.n_dofs());
  solution_u2.reinit(dof_handler.n_dofs());
  prev_solution_u1.reinit(dof_handler.n_dofs());
  prev_solution_u2.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


template <int dim>
void ConvectionDiffusionEquation<dim>::setup_convection
      (const Vector<double>& u1, const Vector<double>& u2)
{
  convection_u1.reinit(solution_u1.size());
  convection_u2.reinit(solution_u1.size());

#ifdef NO_CONVECTION
  return;
#endif

  const unsigned dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double> cell_convection_u1(dofs_per_cell);
  Vector<double> cell_convection_u2(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  QGauss<dim> quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_JxW_values);
  std::vector<double> u1_q_point(quadrature_formula.size());
  std::vector<double> u2_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u1_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u2_q_point(quadrature_formula.size());

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_convection_u1 = 0.;
      cell_convection_u2 = 0.;

      fe_values.reinit(cell);
      fe_values.get_function_values(u1, u1_q_point);
      fe_values.get_function_values(u2, u2_q_point);
      fe_values.get_function_gradients(u1, grad_u1_q_point);
      fe_values.get_function_gradients(u2, grad_u2_q_point);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        double weight = fe_values.JxW(q_index);
        for (const unsigned int i : fe_values.dof_indices())
        {
          cell_convection_u1(i) += weight * fe_values.shape_value(i, q_index) * 
                                  ( grad_u1_q_point[q_index][0] * u1_q_point[q_index] + 
                                    grad_u1_q_point[q_index][1] * u2_q_point[q_index]);
          cell_convection_u2(i) += weight * fe_values.shape_value(i, q_index) * 
                                  ( grad_u2_q_point[q_index][0] * u1_q_point[q_index] + 
                                    grad_u2_q_point[q_index][1] * u2_q_point[q_index]);
        }
      }

      // distribute to global
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
      {
        convection_u1[local_dof_indices[i]] += cell_convection_u1[i];
        convection_u2[local_dof_indices[i]] += cell_convection_u2[i];
      }
    }
}


template <int dim>
void ConvectionDiffusionEquation<dim>::run(){
  const unsigned initial_global_refinement = 1 + level;
  const unsigned n_adaptive_pre_refinement_steps = 4;
  Vector<double> tmp;
  Vector<double> middle_solution_u1;
  Vector<double> middle_solution_u2;
  Vector<double> forcing_term_1;
  Vector<double> forcing_term_2;
  unsigned pre_refinement_step = 0;

  make_mesh();
  triangulation.refine_global(initial_global_refinement);
  setup_system();

start_time_iteration:
  time            = 0.0;
  timestep_number = 0;

  tmp.reinit(solution_u1.size());
  middle_solution_u1.reinit(solution_u1.size());
  middle_solution_u2.reinit(solution_u1.size());
  forcing_term_1.reinit(solution_u1.size());
  forcing_term_2.reinit(solution_u1.size());

  ForcingTerm1<dim> rhs_func_1;
  ForcingTerm2<dim> rhs_func_2;

  TrueSolution1<dim> initial_1;
  TrueSolution2<dim> initial_2;
  initial_1.set_time(0);
  initial_2.set_time(0);
  VectorTools::interpolate(dof_handler,
                           initial_1,
                           prev_solution_u1);
  VectorTools::interpolate(dof_handler,
                           initial_2,
                           prev_solution_u2);
  solution_u1 = prev_solution_u1;
  solution_u2 = prev_solution_u2;
  output_result();

  while(time <= end_time){
    time += time_step;
    timestep_number++;
    std::cout << std::endl << "Time step " << timestep_number << " at t=" << time << std::endl;

    //------------------------------first stage-------------------------------------

    setup_convection(prev_solution_u1, prev_solution_u2);

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(0.5*time_step, laplace_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u1);
    laplace_matrix.vmult(tmp, prev_solution_u1);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-time_step, convection_u1);

    rhs_func_1.set_time(time-time_step);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree+1),
                                        rhs_func_1,
                                        forcing_term_1);
    system_rhs.add(time_step, forcing_term_1);

    constraints.condense(system_matrix, system_rhs);
    solve_time_step(middle_solution_u1);

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(0.5*time_step, laplace_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u2);
    laplace_matrix.vmult(tmp, prev_solution_u2);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-time_step, convection_u2);

    rhs_func_2.set_time(time-time_step);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree+1),
                                        rhs_func_2,
                                        forcing_term_2);
    system_rhs.add(time_step, forcing_term_2);
    
    constraints.condense(system_matrix, system_rhs);
    solve_time_step(middle_solution_u2);


    //------------------------------second stage-------------------------------------

    // compute u** = u_prev + u*
    middle_solution_u1 += prev_solution_u1;
    middle_solution_u2 += prev_solution_u2;
    middle_solution_u1 *= 0.5;
    middle_solution_u2 *= 0.5;

    setup_convection(middle_solution_u1, middle_solution_u2);

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u1);
    laplace_matrix.vmult(tmp, middle_solution_u1);
    system_rhs.add(-time_step, tmp);
    system_rhs.add(-time_step, convection_u1);

    rhs_func_1.set_time(time);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree+1),
                                        rhs_func_1,
                                        tmp);
    system_rhs.add(0.5*time_step, forcing_term_1);
    system_rhs.add(0.5*time_step, tmp);

    constraints.condense(system_matrix, system_rhs);
    solve_time_step(solution_u1);

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u2);
    laplace_matrix.vmult(tmp, middle_solution_u2);
    system_rhs.add(-time_step, tmp);
    system_rhs.add(-time_step, convection_u2);

    rhs_func_2.set_time(time);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree+1),
                                        rhs_func_2,
                                        tmp);
    system_rhs.add(0.5*time_step, forcing_term_2);
    system_rhs.add(0.5*time_step, tmp);

    constraints.condense(system_matrix, system_rhs);
    solve_time_step(solution_u2);
    output_result();

    // Refine mesh 1 times per 5 time-step. At the begining, refine 5 times.
    if ((timestep_number == 1) &&
        (pre_refinement_step < n_adaptive_pre_refinement_steps))
      {
        refine_mesh(initial_global_refinement,
                    initial_global_refinement +
                      n_adaptive_pre_refinement_steps);
        ++pre_refinement_step;
        std::cout << std::endl;
        std::cout << "pre refinement step: " << pre_refinement_step << std::endl;
        std::cout << std::endl;
        goto start_time_iteration;
      }
    else if ((timestep_number > 0) && (timestep_number % 5 == 0))
      {
        refine_mesh(initial_global_refinement,
                    initial_global_refinement +
                      n_adaptive_pre_refinement_steps);
        tmp.reinit(solution_u1.size());
        middle_solution_u1.reinit(solution_u1.size());
        middle_solution_u2.reinit(solution_u1.size());
        forcing_term_1.reinit(solution_u1.size());
        forcing_term_2.reinit(solution_u1.size());
      }

    prev_solution_u1 = solution_u1;
    prev_solution_u2 = solution_u2;
  }
}


int main(int argc, const char *argv[]){
  if(argc != 3){
    std::cerr << "Param error! Please run with command" << std::endl;
    std::cerr << "./convection-diffusion N T" << std::endl;
    std::cerr << "where N is the level of base grid, T is end_time." << std::endl;
    return -1;
  }
  int level = std::stoi(argv[1]);
  double end_time = std::stod(argv[2]);
  ConvectionDiffusionEquation<2> convection_diffusion_equation(level, end_time);
  convection_diffusion_equation.run();
  return 0;
}