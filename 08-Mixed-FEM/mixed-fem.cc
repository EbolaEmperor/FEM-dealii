#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
 
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
 
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
 
using namespace dealii;

#include <fstream>
#include <iostream>

// #define ANALYTIC_SOLUTION

const double alpha = 0.3;
const double beta = 1.;


#ifdef ANALYTIC_SOLUTION

template<int dim>
class ExactSolution : public Function<dim>{
public:
  ExactSolution()
    : Function<dim>(dim + 1)
  {}
  void vector_value(const Point<dim> &p, Vector<double> &value) const override
  {
    AssertDimension(value.size(), dim + 1);
    value[0] = alpha/2 * p[1]*p[1] + beta - alpha/2 * p[0]*p[0];
    value[1] = alpha * p[0]*p[1];
    value[2] = -(alpha/2 * p[0]*p[1]*p[1] + beta * p[0] 
                 - alpha/6 * p[0]*p[0]*p[0]);
  };
};

#endif


template<int dim>
class BoundaryTerm : public Function<dim>{
public:
  double value(const Point<dim> & p, 
               const unsigned int component = 0) const override
  {
    (void)component;
    return -(alpha/2 * p[0]*p[1]*p[1] + beta * p[0] - alpha/6 * p[0]*p[0]*p[0]);
  };
};


template<int dim>
class ForcingTerm : public Function<dim>{
public:
  double value(const Point<dim> & p, 
               const unsigned int component = 0) const override
  {
    (void)p;
    (void)component;
    return 0;
  };
};


template<int dim>
class InverseK : public Function<dim>{
public:
  void value_list(const std::vector<Point<dim>> &points,
                 std::vector<Tensor<2, dim>>   &values) const
  {
    AssertDimension (points.size(), values.size());
#ifdef ANALYTIC_SOLUTION
    (void)points;
    for(auto &value : values)
      value = unit_symmetric_tensor<dim>();
#else
    for (unsigned int p=0; p<points.size(); ++p)
      {
        values[p].clear ();
        const double distance_to_flowline
          = std::fabs(points[p][1]-0.2*std::sin(10*points[p][0]));
        const double permeability = std::max(std::exp(-(distance_to_flowline*
                                                        distance_to_flowline)
                                                      / (0.1 * 0.1)),
                                            0.001);
        for (unsigned int d=0; d<dim; ++d)
          values[p][d][d] = 1./permeability;
      }
#endif
  }
};


template<int dim>
class MixedLaplaceProblem
{
public:
  MixedLaplaceProblem(const int &, const int &);
  void run();

private:
  void make_grid_and_setup();
  void assemble_system();
  void solve();
  void output_results();

  unsigned int degree;
  unsigned int level;

  Triangulation<dim> triangulation;
  FESystem<dim>      fe;
  DoFHandler<dim>    dof_handler;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double> system_rhs;
  BlockVector<double> solution;
};


template<int dim>
MixedLaplaceProblem<dim>::MixedLaplaceProblem
  (const int &degree, const int &level): 
   degree(degree),
   level(level),
   fe(FE_RaviartThomas<dim>(degree), FE_DGQ<dim>(degree)),
   dof_handler(triangulation)
{}


template<int dim>
void MixedLaplaceProblem<dim>::run()
{
  make_grid_and_setup();
  assemble_system();
  solve();
  output_results();
}


template<int dim>
void MixedLaplaceProblem<dim>::make_grid_and_setup()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(level);

  dof_handler.distribute_dofs(fe);
  DoFRenumbering::component_wise(dof_handler);

  auto dofs_per_component = DoFTools::count_dofs_per_fe_component(dof_handler);
  const unsigned n_u = dofs_per_component[0];
  const unsigned n_p = dofs_per_component[dim];

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Total number of cells: " << triangulation.n_cells()
            << std::endl
            << "Number of DoFs: " << dof_handler.n_dofs()
            << " (" << n_u << '+' << n_p << ')' << std::endl;
  
  const std::vector<types::global_dof_index> block_size{n_u, n_p};
  BlockDynamicSparsityPattern dsp(block_size, block_size);
  DoFTools::make_sparsity_pattern(dof_handler, dsp);

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  solution.reinit(block_size);
  system_rhs.reinit(block_size);
}


template<int dim>
void MixedLaplaceProblem<dim>::assemble_system()
{
  QGauss<dim> quadrature(degree+2);
  QGauss<dim-1> face_quadrature(degree+2);

  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_values | update_gradients |
                          update_JxW_values | update_quadrature_points);
  FEFaceValues<dim> fe_face_values(fe,
                          face_quadrature,
                          update_values | update_normal_vectors |
                          update_JxW_values | update_quadrature_points);
  
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
 
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  FEValuesExtractors::Vector velocities(0);
  FEValuesExtractors::Scalar pressure(dim);

  InverseK<dim> inverse_k_function;
  std::vector<Tensor<2, dim>> k_inv;
  k_inv.resize(quadrature.size());

  ForcingTerm<dim> f;
  BoundaryTerm<dim> g;

  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0.;
    cell_rhs    = 0.;
    const auto &q_points = fe_values.get_quadrature_points();
    inverse_k_function.value_list(q_points, k_inv);

    for(const unsigned q : fe_values.quadrature_point_indices())
    {
      double jxw = fe_values.JxW(q);
      for(const unsigned i : fe_values.dof_indices())
      {
        auto velovity_i = fe_values[velocities].value(i,q);
        auto pressure_i = fe_values[pressure].value(i,q);
        auto div_velocity_i = fe_values[velocities].divergence(i,q);

        for(const unsigned j : fe_values.dof_indices())
        {
          auto velovity_j = fe_values[velocities].value(j,q);
          auto pressure_j = fe_values[pressure].value(j,q);
          auto div_velocity_j = fe_values[velocities].divergence(j,q);

          cell_matrix(i,j) += jxw * 
            (  velovity_i * k_inv[q] * velovity_j
             - div_velocity_i * pressure_j
             - pressure_i * div_velocity_j );
        }
        cell_rhs(i) -= jxw * pressure_i * f.value(q_points[q]);
      }
    }

    for(const auto &f : cell->face_iterators())
      if(f->at_boundary())
      {
        fe_face_values.reinit(cell, f);
        auto q_face_points = fe_face_values.get_quadrature_points();

        for(const unsigned q : fe_face_values.quadrature_point_indices())
        {
          auto normal = fe_face_values.normal_vector(q);
          double jxw = fe_face_values.JxW(q);
          for(const unsigned i : fe_face_values.dof_indices())
          {
            cell_rhs(i) -= jxw * normal * 
                           fe_face_values[velocities].value(i,q) * 
                           g.value(q_face_points[q]);
          }
        }
      }
    
    cell->get_dof_indices(local_dof_indices);
    for(const unsigned i : fe_values.dof_indices())
    {
      for(const unsigned j : fe_values.dof_indices())
        system_matrix.add(local_dof_indices[i], 
                          local_dof_indices[j],
                          cell_matrix(i, j));
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  }
}


template<int dim>
void MixedLaplaceProblem<dim>::solve()
{
  const auto &M = system_matrix.block(0,0);
  const auto &B = system_matrix.block(0,1);

  const auto &F = system_rhs.block(0);
  const auto &G = system_rhs.block(1);

  auto &U = solution.block(0);
  auto &P = solution.block(1);

  const auto op_M = linear_operator(M);
  const auto op_B = linear_operator(B);

  ReductionControl         reduction_control(2000, 2.3e-16, 1e-10);
  SolverCG<Vector<double>> solver_M(reduction_control);
  PreconditionJacobi<SparseMatrix<double>> preconditioner_M;
  preconditioner_M.initialize(M);

  const auto op_M_inv = inverse_operator(op_M, solver_M, preconditioner_M);
  const auto op_S = transpose_operator(op_B) * op_M_inv * op_B;
  const auto op_aS = 
    transpose_operator(op_B) * linear_operator(preconditioner_M) * op_B;
  
  IterationNumberControl   iteration_number_control(30, 2.3e-16);
  SolverCG<Vector<double>> solver_aS(iteration_number_control);

  const auto preconditioner_S = 
    inverse_operator(op_aS, solver_aS, PreconditionIdentity());
  
  const auto schur_rhs = transpose_operator(op_B) * op_M_inv * F - G;

  SolverControl            solver_control_S(2000, 1e-12);
  SolverCG<Vector<double>> solver_S(solver_control_S);

  const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);
  P = op_S_inv * schur_rhs;
  std::cout << solver_control_S.last_step()
            << " CG Schur complement iterations to obtain convergence."
            << std::endl;
  U = op_M_inv * (F - op_B * P);
}


template <int dim>
void MixedLaplaceProblem<dim>::output_results()
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

  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);

#ifdef ANALYTIC_SOLUTION
  const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
  const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                    dim + 1);

  ExactSolution<dim> exact_solution;
  Vector<double> cellwise_errors(triangulation.n_active_cells());

  QTrapezoid<1>  q_trapez;
  QIterated<dim> quadrature(q_trapez, degree + 2);

  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    cellwise_errors,
                                    quadrature,
                                    VectorTools::L2_norm,
                                    &pressure_mask);
  const double p_l2_error =
    VectorTools::compute_global_error(triangulation,
                                      cellwise_errors,
                                      VectorTools::L2_norm);

  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    cellwise_errors,
                                    quadrature,
                                    VectorTools::L2_norm,
                                    &velocity_mask);
  const double u_l2_error =
    VectorTools::compute_global_error(triangulation,
                                      cellwise_errors,
                                      VectorTools::L2_norm);

  std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
            << ",   ||e_u||_L2 = " << u_l2_error << std::endl;
#endif
}


int main(int argc, const char *argv[]){
  if(argc != 3)
  {
    std::cerr << "Param error!" << std::endl;
    return -1;
  }
  MixedLaplaceProblem<2> laplace(std::stoi(argv[1]),  // degree of FE
                                 std::stoi(argv[2])); // grid size
  laplace.run();
  return 0;
}