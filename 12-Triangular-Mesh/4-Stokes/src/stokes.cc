#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor_function.h>
 
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
 
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
#include <deal.II/fe/mapping_fe.h>
 
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
 
#include <iostream>
#include <fstream>
#include <memory>
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


class PreconditionStokes : public Subscriptor
{
public:
  void initialize(const BlockSparseMatrix<double>& system_matrix,
                  const BlockSparsityPattern& preconditioner_sparsity_pattern,
                  const BlockSparseMatrix<double>& preconditioner_matrix,
                  const std::vector<types::global_dof_index>& dofs_per_block_input);
  void vmult(Vector<double> &dst, const Vector<double> &src) const;

private:
  TrilinosWrappers::PreconditionAMG preconditioner[2];
  SparseMatrix<double> preconditioner_pressure;
  std::vector<types::global_dof_index> dofs_per_block;
};


void PreconditionStokes::initialize(
    const BlockSparseMatrix<double>& system_matrix,
    const BlockSparsityPattern& preconditioner_sparsity_pattern,
    const BlockSparseMatrix<double>& preconditioner_matrix,
    const std::vector<types::global_dof_index>& dofs_per_block_input)
{
  dofs_per_block = dofs_per_block_input;
  preconditioner[0].initialize(system_matrix.block(0,0));
  preconditioner[1].initialize(system_matrix.block(1,1));

  preconditioner_pressure.reinit(preconditioner_sparsity_pattern.block(2,2));
  preconditioner_pressure.copy_from(preconditioner_matrix.block(2,2));
  for(auto &p : preconditioner_pressure)
    p.value() = 1./p.value();
}


void PreconditionStokes::vmult(Vector<double> &dst, const Vector<double> &src) const
{
  BlockVector<double> block_src;
  BlockVector<double> block_dst;
  block_src.reinit(dofs_per_block);
  block_dst.reinit(dofs_per_block);

  block_src = src;
  preconditioner[0].vmult(block_dst.block(0), block_src.block(0));
  preconditioner[1].vmult(block_dst.block(1), block_src.block(1));
  preconditioner_pressure.vmult(block_dst.block(2), block_src.block(2));
  dst = block_dst;
}


template <int dim>
class StokesProblem
{
public:
  StokesProblem(const unsigned int degree);
  void run();

private:
  void setup_dofs();
  void assemble_system();
  void solve();
  void output_results(const unsigned int refinement_cycle) const;
  void refine_mesh();
  void compute_L2_error();

  const unsigned int degree;

  Triangulation<dim> triangulation;
  FESystem<dim>      fe;
  DoFHandler<dim>    dof_handler;

  std::vector<types::global_dof_index> dofs_per_block;

  AffineConstraints<double> constraints;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparsityPattern      preconditioner_sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;
  BlockSparseMatrix<double> preconditioner_matrix;

  BlockVector<double> solution;
  Vector<double> full_solution;
  Vector<double> full_system_rhs;

  SparsityPattern      full_sparsity_pattern;
  SparseMatrix<double> full_system_matrix;
};



template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues()
    : Function<dim>(dim + 1)
  {}

  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;

  virtual void vector_value(const Point<dim> &p,
                            Vector<double>   &value) const override;
};


template <int dim>
double BoundaryValues<dim>::value(const Point<dim>  &p,
                                  const unsigned int component) const
{
  Assert(component < this->n_components,
          ExcIndexRange(component, 0, this->n_components));
  (void)p;
  (void)component;
  return 0;
}


template <int dim>
void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                        Vector<double>   &values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryValues<dim>::value(p, c);
}


template <int dim>
class RightHandSide : public TensorFunction<1, dim>
{
public:
  RightHandSide()
    : TensorFunction<1, dim>()
  {}

  virtual Tensor<1, dim> value(const Point<dim> &p) const override;

  virtual void value_list(const std::vector<Point<dim>> &p,
                          std::vector<Tensor<1, dim>> &value) const override;
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
void RightHandSide<dim>::value_list(const std::vector<Point<dim>> &vp,
                                    std::vector<Tensor<1, dim>> &values) const
{
  for (unsigned int c = 0; c < vp.size(); ++c)
    {
      values[c] = RightHandSide<dim>::value(vp[c]);
    }
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


template <int dim>
StokesProblem<dim>::StokesProblem(const unsigned int degree)
  : degree(degree)
  , fe(FE_SimplexP<dim>(degree + 1) ^ dim, FE_SimplexP<dim>(degree))
  , dof_handler(triangulation)
{}


template <int dim>
void StokesProblem<dim>::setup_dofs()
{
  full_system_matrix.clear();
  preconditioner_matrix.clear();

  dof_handler.distribute_dofs(fe);

  // auto dof_location_map =
  //   DoFTools::map_dofs_to_support_points(MappingFE<2>(fe), dof_handler);

  DoFRenumbering::Cuthill_McKee(dof_handler);
  DoFRenumbering::component_wise(dof_handler);

  {
    constraints.clear();

    const FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                              0,
                                              BoundaryValues<dim>(),
                                              constraints,
                                              fe.component_mask(velocities));
  }

  constraints.close();

  dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler);

  {
    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
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

  {
    DynamicSparsityPattern fulldsp(dof_handler.n_dofs(),
                                dof_handler.n_dofs());
    int idx_r=0, idx_c=0;
    for (unsigned int c = 0; c < dim + 1; ++c){
      idx_c = 0;
      for (unsigned int d = 0; d < dim + 1; ++d){
        SparsityPattern subsp;
        subsp.copy_from(sparsity_pattern.block(c,d));
        for(const auto p : subsp){
          fulldsp.add(p.row()+idx_r, p.column()+idx_c);
        }
        idx_c += dofs_per_block[d];
      }
      idx_r += dofs_per_block[c];
    }

    full_sparsity_pattern.copy_from(fulldsp);
    // std::ofstream out("sparsity-pattern.svg");
    // full_sparsity_pattern.print_svg(out);
  }

  {
    BlockDynamicSparsityPattern preconditioner_dsp(dofs_per_block,
                                                    dofs_per_block);

    Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        preconditioner_coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern(dof_handler,
                                    preconditioner_coupling,
                                    preconditioner_dsp,
                                    constraints,
                                    false);

    preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
  }

  full_system_matrix.reinit(full_sparsity_pattern);
  preconditioner_matrix.reinit(preconditioner_sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  full_system_rhs.reinit(dof_handler.n_dofs());
  full_solution.reinit(dof_handler.n_dofs());
  solution.reinit(dofs_per_block);
}



template <int dim>
void StokesProblem<dim>::assemble_system()
{
  full_system_matrix    = 0;
  full_system_rhs       = 0;
  preconditioner_matrix = 0;

  QGaussSimplex<dim> quadrature_formula(degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                  dofs_per_cell);
  Vector<double>     local_rhs(dofs_per_cell);

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
      local_matrix                = 0;
      local_preconditioner_matrix = 0;
      local_rhs                   = 0;

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
              local_preconditioner_matrix(i, i) +=
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
                                              full_system_matrix,
                                              full_system_rhs);
      constraints.distribute_local_to_global(local_matrix,
                                              local_dof_indices,
                                              system_matrix);
      constraints.distribute_local_to_global(local_preconditioner_matrix,
                                              local_dof_indices,
                                              preconditioner_matrix);
    }
}


template <int dim>
void StokesProblem<dim>::solve()
{
  SolverControl solver_control(50000, 1e-6 * full_system_rhs.l2_norm());
  SolverMinRes<Vector<double>> minres(solver_control);

  PreconditionStokes preconditioner;
  preconditioner.initialize(system_matrix,
                            preconditioner_sparsity_pattern,
                            preconditioner_matrix,
                            dofs_per_block);

  minres.solve(full_system_matrix, 
               full_solution, 
               full_system_rhs, 
               preconditioner);
  solution = full_solution;
  std::cout << "      " << solver_control.last_step()
            << " MINRES iterations." << std::endl;

  constraints.distribute(solution);
  const double mean_value = solution.block(dim).mean_value();
  solution.block(dim).add(-mean_value);
}


template <int dim>
void
StokesProblem<dim>::output_results(const unsigned int refinement_cycle) const
{
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("pressure");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,
                            solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);
  data_out.build_patches();

  std::ofstream output(
    "solution/sol-" + Utilities::int_to_string(refinement_cycle, 2) + ".vtk");
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
  
  std::cout << "      u1 error: " << sqrt(err_u1) << std::endl;
  std::cout << "      u2 error: " << sqrt(err_u2) << std::endl;
  std::cout << "      p  error: " << sqrt(err_p)  << std::endl << std::flush;
}


template <int dim>
void StokesProblem<dim>::run()
{
  GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, 1);
  triangulation.refine_global(2);
  CPUTimer timer;

  for (unsigned int refinement_cycle = 0; refinement_cycle < 9;
        ++refinement_cycle)
    {
      std::cout << "Level " << 2+refinement_cycle;
      std::cout << " ----------------------------------------" << std::endl;
      if (refinement_cycle > 0) triangulation.refine_global(1);
      setup_dofs();

      timer.reset();
      std::cout << "   Assembling...  " << std::flush;
      assemble_system();
      std::cout << timer() << "s" << std::endl;

      timer.reset();
      std::cout << "   Solving..." << std::endl << std::flush;
      solve();
      std::cout << "   Solved in " << timer() << "s" << std::endl;

      std::cout << "   Outputing...  " << std::endl << std::flush;
      output_results(refinement_cycle);

      std::cout << "   Computing L2 error...  " << std::endl << std::flush;
      compute_L2_error();
      std::cout << std::endl << std::flush;
    }
}
 
 
int main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      StokesProblem<2> flow_problem(1);
      flow_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
 
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
 
  return 0;
}