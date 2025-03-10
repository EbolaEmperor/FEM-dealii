/**********************************************************
 * This is a solver for Stokes system
 * Spacial Discretization: (P2-P1) mixed elements.
 * Linear solver: Inexact Uzawa iteration
 *                    u <- inv(A) * (f - B' * p)
 *                    p <- p + inv(M_Q) * B * u
 *                replace inv(A) with an AMG V-cycle for A
 *                replace inv(M_Q) with inv(diag(M_Q))
***********************************************************/

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

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

#include <iostream>
#include <fstream>
#include <memory>


//-------------------------------Solver class---------------------------------

template<int dim>
class StokesProblem
{
public:
  StokesProblem(const int &, const int &);
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void compute_L2_error();
  void output_results(const unsigned &);

  unsigned int degree;
  unsigned int level;

  Triangulation<dim> triangulation;
  FESystem<dim>      fe;
  DoFHandler<dim>    dof_handler;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double> system_rhs;
  BlockVector<double> solution;
  BlockVector<double> checker;
  BlockVector<double> block_diag_MQ;

  AffineConstraints<double> constraints;

  std::shared_ptr<TrilinosWrappers::PreconditionAMG> A_preconditioner;
};


//--------------------------RHS and True Solution---------------------------

template <int dim>
class RightHandSide : public TensorFunction<1, dim>
{
public:
  RightHandSide()
    : TensorFunction<1, dim>()
  {}
  virtual Tensor<1, dim> value(const Point<dim> &p) const override;
};


template <int dim>
Tensor<1, dim> RightHandSide<dim>::value(const Point<dim> &p) const
{
  Tensor<1, dim> ans;
  ans[0] = M_PI * sin(M_PI*p[0]) * sin(M_PI*p[1]) 
            - 2*M_PI*M_PI*M_PI * (-1. + 2.*cos(2*M_PI*p[0])) * sin(2*M_PI*p[1]);
  ans[1] = -M_PI*cos(M_PI*p[0])*cos(M_PI*p[1]) 
            + 2*M_PI*M_PI*M_PI * (-1. + 2.*cos(2*M_PI*p[1])) * sin(2*M_PI*p[0]);
  return ans;
}


template <int dim>
class TrueSolution : public Function<dim>
{
public:
  TrueSolution()
    : Function<dim>()
  {}
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};


template <int dim>
double TrueSolution<dim>::value(const Point<dim> &p,
                                const unsigned int component) const
{
  if(component == 0)
    return M_PI * pow(sin(M_PI*p[0]), 2.) * sin(2.*M_PI*p[1]);
  else if(component == 1)
    return -M_PI * pow(sin(M_PI*p[1]), 2.) * sin(2.*M_PI*p[0]);
  else if(component == 2)
    return -cos(M_PI*p[0]) * sin(M_PI*p[1]);
  return 0;
}


//---------------------------Solver implemention------------------------------

template<int dim>
StokesProblem<dim>::StokesProblem
  (const int &degree, const int &level): 
   degree(degree),
   level(level),
   triangulation(),
   fe(FE_SimplexP<dim>(degree+1) ^ dim, FE_SimplexP<dim>(degree)),
   dof_handler(triangulation)
{}


template<int dim>
void StokesProblem<dim>::run()
{
  make_grid();
  CPUTimer timer;
  for(unsigned cycle = 0; cycle+2 <= level; cycle++)
  {
    std::cout << "Level: " << cycle+2 << "----------------------------------" << std::endl;
    if(cycle) triangulation.refine_global(1);
    std::cout << "Setup..." << std::endl;
    setup_system();
    timer.reset();
    std::cout << "Assembling..." << std::endl;
    assemble_system();
    std::cout << "  " << timer() << "s." << std::endl;
    timer.reset();
    std::cout << "Solving..." << std::endl;
    solve();
    std::cout << "  " << timer() << "s." << std::endl;
    std::cout << "Output results..." << std::endl;
    output_results(cycle);
    std::cout << "Computing error..." << std::endl;
    compute_L2_error();
    std::cout << std::endl;
  }
}


template<int dim>
void StokesProblem<dim>::make_grid()
{
  GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, 1);
  triangulation.refine_global(2);
}


template<int dim>
void StokesProblem<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dof_handler);

  std::vector<unsigned> block_component(dim+1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(dof_handler, block_component);

  const FEValuesExtractors::Vector velocities(0);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints,
                                           fe.component_mask(velocities));
  constraints.close();

  auto dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler,
                                                          block_component);
  const unsigned n_u = dofs_per_block[0];
  const unsigned n_p = dofs_per_block[1];

  std::cout << "  Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "  Total number of cells: " << triangulation.n_cells()
            << std::endl
            << "  Number of DoFs: " << dof_handler.n_dofs()
            << " (" << n_u << '+' << n_p << ')' << std::endl;
  {
    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

    Table<2, DoFTools::Coupling> coupling(dim+1, dim+1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if ((c==d) ^ (c==dim || d==dim))
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;
    
    DoFTools::make_sparsity_pattern(
      dof_handler, coupling, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);
  }
  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);
  checker.reinit(dofs_per_block);
  block_diag_MQ.reinit(dofs_per_block);
}


template<int dim>
void StokesProblem<dim>::assemble_system()
{
  system_matrix   = 0;
  system_rhs      = 0;
  block_diag_MQ   = 0;

  QGaussSimplex<dim> quadrature_formula(degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     local_rhs(dofs_per_cell);
  Vector<double>     local_diag_MQ(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const RightHandSide<dim>    right_hand_side;
  std::vector<Tensor<1, dim>> rhs_values(n_q_points, Tensor<1, dim>());

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<2, dim>>          grad_phi_u(dofs_per_cell);
  std::vector<double>                  div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>>          phi_u(dofs_per_cell);
  std::vector<double>                  phi_p(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      local_matrix    = 0;
      local_diag_MQ   = 0;
      local_rhs       = 0;

      right_hand_side.value_list(fe_values.get_quadrature_points(),
                                  rhs_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              grad_phi_u[k] = fe_values[velocities].gradient(k, q);
              div_phi_u[k]  = fe_values[velocities].divergence(k, q);
              phi_u[k]      = fe_values[velocities].value(k, q);
              phi_p[k]      = fe_values[pressure].value(k, q);
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j <= i; ++j)
                {
                  double gg = grad_phi_u[i][0] * grad_phi_u[j][0]
                            + grad_phi_u[i][1] * grad_phi_u[j][1];
                  local_matrix(i, j) +=
                    ( gg                                        // (1)
                      - div_phi_u[i] * phi_p[j]                 // (2)
                      - phi_p[i] * div_phi_u[j])                // (3)
                    * fe_values.JxW(q);                        // * dx
                }
              local_rhs(i) += phi_u[i]            // phi_u_i(x_q)
                              * rhs_values[q]     // * f(x_q)
                              * fe_values.JxW(q); // * dx
              local_diag_MQ(i) +=
                    (phi_p[i] * phi_p[i]) // (4)
                    * fe_values.JxW(q);   // * dx
            }
        }

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
            local_matrix(i, j) = local_matrix(j, i);

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(local_matrix,
                                              local_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
      constraints.distribute_local_to_global(local_diag_MQ,
                                              local_dof_indices,
                                              block_diag_MQ);
    }
}


template<int dim>
void StokesProblem<dim>::solve()
{
  // Build preconditioners
  TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
  additional_data.higher_order_elements = true;
  additional_data.w_cycle = true;

  A_preconditioner.reset();
  A_preconditioner = 
    std::make_shared<TrilinosWrappers::PreconditionAMG>();
  A_preconditioner->initialize(system_matrix.block(0,0), additional_data);

  auto diag_MQ_inverse = block_diag_MQ.block(1);
  for(auto& p : diag_MQ_inverse) p = 1. / p;

  // Uzawa iteration
  Vector<double> rhs(solution.block(0).size());
  Vector<double> correction_u(solution.block(0).size());
  Vector<double> correction_p(solution.block(1).size());
  int cnt = 0;

  const auto B      = linear_operator(system_matrix.block(1,0));
  const auto BT     = linear_operator(system_matrix.block(0,1));
  const auto A      = linear_operator(system_matrix.block(0,0));
  const auto & f = system_rhs.block(0);
  auto & u = solution.block(0);
  auto & p = solution.block(1);

  do{
    cnt++;
    rhs = f - BT * p - A * u;
    A_preconditioner->vmult(correction_u, rhs);
    u += correction_u;
    constraints.distribute(solution);

    correction_p = B * u;
    for(unsigned i = 0; i < p.size(); i++)
      correction_p[i] *= diag_MQ_inverse[i];
    p += correction_p;

    system_matrix.vmult(checker, solution);
    checker.add(-1, system_rhs);
  }while(checker.l2_norm() >= 1e-7*system_rhs.l2_norm());

  std::cout << "  " << cnt << " Uzawa iterations." << std::endl;
}


template <int dim>
void StokesProblem<dim>::output_results(const unsigned &cycle)
{
  std::vector<std::string> solution_names(dim, "u");
  solution_names.emplace_back("p");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation(dim,
                    DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler,
                            solution,
                            solution_names,
                            interpretation);

  data_out.build_patches(degree + 1);

  std::ofstream output(
    "solution/sol-" + Utilities::int_to_string(cycle, 2) + ".vtk");
  data_out.write_vtk(output);
}


template <int dim>
void StokesProblem<dim>::compute_L2_error()
{
  QGaussSimplex<dim> quadrature_formula(degree + 2);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  const unsigned int n_q_points = quadrature_formula.size();

  const TrueSolution<dim> true_solution;
  std::vector<Tensor<1, dim>> numerical_results_u(n_q_points);
  std::vector<double>         numerical_results_p(n_q_points);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  double err_u1 = 0.;
  double err_u2 = 0.;
  double err_p  = 0.;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values[velocities].get_function_values(solution, numerical_results_u);
      fe_values[pressure].get_function_values(solution, numerical_results_p);
      auto q_points = fe_values.get_quadrature_points();

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          auto jxw = fe_values.JxW(q);
          err_u1 += pow(true_solution.value(q_points[q], 0) - numerical_results_u[q][0], 2.) * jxw;
          err_u2 += pow(true_solution.value(q_points[q], 1) - numerical_results_u[q][1], 2.) * jxw;
          err_p  += pow(true_solution.value(q_points[q], 2) - numerical_results_p[q], 2.) * jxw;
        }
    }
  
  std::cout << "  u1 error: " << sqrt(err_u1) << std::endl;
  std::cout << "  u2 error: " << sqrt(err_u2) << std::endl;
  std::cout << "  p  error: " << sqrt(err_p)  << std::endl << std::flush;
}


int main(int argc, char *argv[]){
  if(argc != 3)
  {
    std::cerr << "Param error! Please run with command" << std::endl;
    std::cerr << "./stokes k N" << std::endl;
    std::cerr << "where k is the degree of FE, and N is the finest grid size."
              << std::endl;
    return -1;
  }
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  StokesProblem<2> stokes(std::stoi(argv[1]),  // degree of FE
                          std::stoi(argv[2])); // grid size
  stokes.run();
  return 0;
}