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
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

using namespace dealii;

#include <iostream>
#include <fstream>
#include <memory>

//--------------------------Inner Preconditioners-----------------------------

template<int dim>
struct InnerPreconditioner;

template<>
struct InnerPreconditioner<2>
{
  using type = SparseDirectUMFPACK;
};

template<>
struct InnerPreconditioner<3>
{
  using type = SparseILU<double>;
};


//-------------------------------Solver class---------------------------------

template<int dim>
class StokesProblem
{
public:
  StokesProblem(const int &, const int &);
  void run();

private:
  void make_grid();
  void refine_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(const unsigned &);

  unsigned int degree;
  unsigned int level;

  Triangulation<dim> triangulation;
  FESystem<dim>      fe;
  DoFHandler<dim>    dof_handler;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparsityPattern      preconditioner_sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;
  BlockSparseMatrix<double> preconditioner_matrix;

  BlockVector<double> system_rhs;
  BlockVector<double> solution;

  AffineConstraints<double> constraints;

  std::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;
};


//--------------------------Dirichlet Boundary Term---------------------------

template <int dim>
class BoundaryTerm : public Function<dim>
{
public:
  BoundaryTerm()
    : Function<dim>(dim + 1)
  {}
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
  virtual void vector_value(const Point<dim> &p,
                            Vector<double>   &value) const override;
};

template <int dim>
double BoundaryTerm<dim>::value(const Point<dim>  &p,
                                  const unsigned int component) const
{
  Assert(component < this->n_components,
          ExcIndexRange(component, 0, this->n_components));
  if (component == 0)
  {
    if (p[0] < 0) return -1;
    else if (p[0] > 0) return 1;
    else return 0;
  }
  return 0;
}

template <int dim>
void BoundaryTerm<dim>::vector_value(const Point<dim> &p,
                                        Vector<double>   &values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryTerm<dim>::value(p, c);
}


//-------------------------------Forcing Term---------------------------------

template <int dim>
class ForcingTerm : public TensorFunction<1, dim>
{
public:
  ForcingTerm()
    : TensorFunction<1, dim>()
  {}
  virtual Tensor<1, dim> value(const Point<dim> &p) const override;
  virtual void value_list(const std::vector<Point<dim>> &p,
                          std::vector<Tensor<1, dim>> &value) const override;
};

template <int dim>
Tensor<1, dim> ForcingTerm<dim>::value(const Point<dim> & /*p*/) const
{
  return Tensor<1, dim>();
}

template <int dim>
void ForcingTerm<dim>::value_list(const std::vector<Point<dim>> &vp,
                                    std::vector<Tensor<1, dim>> &values) const
{
  for (unsigned int c = 0; c < vp.size(); ++c)
      values[c] = ForcingTerm<dim>::value(vp[c]);
}


//----------------------------Inverse matrices--------------------------------

template <class MatrixType, class PreconditionerType>
class InverseMatrix : public Subscriptor
{
public:
  InverseMatrix(const MatrixType &,
                const PreconditionerType &);
  void vmult(Vector<double> &dst, const Vector<double> &src) const;

private:
  const SmartPointer<const MatrixType>         matrix;
  const SmartPointer<const PreconditionerType> preconditioner;
};

template <class MatrixType, class PreconditionerType>
InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix
  (const MatrixType &m, const PreconditionerType &p)
  : matrix(&m), preconditioner(&p)
{}

template <class MatrixType, class PreconditionerType>
void InverseMatrix<MatrixType, PreconditionerType>::vmult
  (Vector<double> &dst, const Vector<double> &src) const
{
  SolverControl control(src.size(), 1e-6*src.l2_norm());
  SolverCG<Vector<double>> solver(control);
  dst = 0.;
  solver.solve(*matrix, dst, src, *preconditioner); 
}


//----------------------------Schur complement--------------------------------

template<class PreconditionerType>
class SchurComplement : public Subscriptor
{
public:
  SchurComplement(
    const BlockSparseMatrix<double> &system_matrix,
    const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse);
  void vmult(Vector<double> &dst, const Vector<double> &src) const;

private:
  const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
  const SmartPointer<
    const InverseMatrix<SparseMatrix<double>, PreconditionerType>>
    A_inverse;
  
  mutable Vector<double> tmp1, tmp2;
};

template<class PreconditionerType>
SchurComplement<PreconditionerType>::SchurComplement(
  const BlockSparseMatrix<double> &system_matrix,
  const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse)
  : system_matrix(&system_matrix),
    A_inverse(&A_inverse),
    tmp1(system_matrix.block(0,0).m()),
    tmp2(system_matrix.block(0,0).m())
{}

template<class PreconditionerType>
void SchurComplement<PreconditionerType>::vmult
  (Vector<double> &dst, const Vector<double> &src) const
{
  system_matrix->block(0,1).vmult(tmp1, src);
  A_inverse->vmult(tmp2, tmp1);
  system_matrix->block(1,0).vmult(dst,tmp2);
}


//---------------------------Solver implemention------------------------------

template<int dim>
StokesProblem<dim>::StokesProblem
  (const int &degree, const int &level): 
   degree(degree),
   level(level),
   triangulation(Triangulation<dim>::maximum_smoothing),
   fe(FE_Q<dim>(degree+1) ^ dim, FE_Q<dim>(degree)),
   dof_handler(triangulation)
{}


template<int dim>
void StokesProblem<dim>::run()
{
  make_grid();
  for(unsigned cycle = 0; cycle < level; cycle++)
  {
    std::cout << "Refinement cycle: " << cycle << std::endl;
    if(cycle) refine_grid();
    std::cout << "Setup..." << std::endl;
    setup_system();
    std::cout << "Assembling..." << std::endl;
    assemble_system();
    std::cout << "Solving..." << std::endl;
    solve();
    std::cout << "Output results..." << std::endl << std::endl;
    output_results(cycle);
  }
}


template<int dim>
void StokesProblem<dim>::make_grid()
{
  std::vector<unsigned> subdivisions(dim, 1);
  subdivisions[0] = 4;

  const Point<dim> bottom_left = (dim == 2 ?                
                                    Point<dim>(-2, -1) :    // 2d case
                                    Point<dim>(-2, 0, -1)); // 3d case
  const Point<dim> top_right = (dim == 2 ?              
                                  Point<dim>(2, 0) :    // 2d case
                                  Point<dim>(2, 1, 0)); // 3d case

  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            subdivisions,
                                            bottom_left,
                                            top_right);
  
  for(const auto& cell : triangulation.active_cell_iterators())
    for(const auto& face : cell->face_iterators())
      if(face->center()[dim-1] == 0)
        face->set_all_boundary_ids(1);
  
  triangulation.refine_global(4-dim);
}


template<int dim>
void StokesProblem<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

  const FEValuesExtractors::Scalar pressure(dim);
  KellyErrorEstimator<dim>::estimate(
    dof_handler,
    QGauss<dim-1>(degree+1),
    std::map<types::boundary_id, const Function<dim> *>(),
    solution,
    estimated_error_per_cell,
    fe.component_mask(pressure));
  
  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.3,
                                                  0.);
  triangulation.execute_coarsening_and_refinement();
}


template<int dim>
void StokesProblem<dim>::setup_system()
{
  A_preconditioner.reset();
  system_matrix.clear();
  preconditioner_matrix.clear();

  dof_handler.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dof_handler);

  std::vector<unsigned> block_component(dim+1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(dof_handler, block_component);

  const FEValuesExtractors::Vector velocities(0);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           BoundaryTerm<dim>(),
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
    for(unsigned i = 0; i <= dim; i++)
      for(unsigned j = 0; j <= dim; j++)
        coupling[i][j] = (i==dim && j==dim) ? 
                         DoFTools::none : DoFTools::always;
    
    DoFTools::make_sparsity_pattern(
      dof_handler, coupling, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);
  }
  {
    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

    Table<2, DoFTools::Coupling> coupling(dim+1, dim+1);
    for(unsigned i = 0; i <= dim; i++)
      for(unsigned j = 0; j <= dim; j++)
        coupling[i][j] = (i==dim && j==dim) ? 
                         DoFTools::always : DoFTools::none;
    
    DoFTools::make_sparsity_pattern(
      dof_handler, coupling, dsp, constraints, false);
    preconditioner_sparsity_pattern.copy_from(dsp);
  }

  system_matrix.reinit(sparsity_pattern);
  preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

  solution.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);
}


template<int dim>
void StokesProblem<dim>::assemble_system()
{
  system_matrix         = 0.;
  system_rhs            = 0.;
  preconditioner_matrix = 0.;

  QGauss<dim> quadrature(degree+2);
  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_values | update_gradients | 
                          update_JxW_values | update_quadrature_points);
  
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
 
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> preconditioner_cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  FEValuesExtractors::Vector velocities(0);
  FEValuesExtractors::Scalar pressure(dim);

  ForcingTerm<dim> f;

  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix                = 0.;
    preconditioner_cell_matrix = 0.;
    cell_rhs                   = 0.;
    const auto &q_points = fe_values.get_quadrature_points();

    for(const unsigned q : fe_values.quadrature_point_indices())
    {
      double jxw = fe_values.JxW(q);
      for(unsigned i = 0; i < dofs_per_cell; ++i)
      {
        auto velocity_i     = fe_values[velocities].value(i,q);
        auto sg_velovity_i  = fe_values[velocities].symmetric_gradient(i,q);
        auto pressure_i     = fe_values[pressure].value(i,q);
        auto div_velocity_i = fe_values[velocities].divergence(i,q);

        for(unsigned j = 0; j <= i; j++)
        {
          auto sg_velovity_j  = fe_values[velocities].symmetric_gradient(j,q);
          auto pressure_j     = fe_values[pressure].value(j,q);
          auto div_velocity_j = fe_values[velocities].divergence(j,q);

          cell_matrix(i,j) += jxw * 
            (  2 * (sg_velovity_i * sg_velovity_j)
             - div_velocity_i * pressure_j
             - pressure_i * div_velocity_j );
          preconditioner_cell_matrix(i,j) += jxw * pressure_i * pressure_j;
        }
        cell_rhs(i) += jxw * velocity_i * f.value(q_points[q]);
      }
    }

    // Fill the symmetric upper-triangular part
    for(unsigned i = 0; i < dofs_per_cell; ++i)
      for(unsigned j = i+1; j < dofs_per_cell; ++j)
      {
        cell_matrix(i,j) = cell_matrix(j,i);
        preconditioner_cell_matrix(i,j) = preconditioner_cell_matrix(j,i);
      }
    
    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(cell_matrix,
                                           cell_rhs,
                                           local_dof_indices,
                                           system_matrix,
                                           system_rhs);
    constraints.distribute_local_to_global(preconditioner_cell_matrix,
                                           local_dof_indices,
                                           preconditioner_matrix);
  }

  std::cout << "  Computing inner preconditioner..." << std::endl;
  A_preconditioner = 
    std::make_shared<typename InnerPreconditioner<dim>::type>();
  A_preconditioner->initialize(
    system_matrix.block(0,0),
    typename InnerPreconditioner<dim>::type::AdditionalData());
}


template<int dim>
void StokesProblem<dim>::solve()
{
  // Build inner solver
  const InverseMatrix<SparseMatrix<double>,
                      typename InnerPreconditioner<dim>::type>
              A_inverse(system_matrix.block(0,0), *A_preconditioner);
  
  Vector<double> tmp(solution.block(0).size());
  
  // Compute rhs of Schur complement
  Vector<double> schur_rhs(solution.block(1).size());
  A_inverse.vmult(tmp, system_rhs.block(0));
  system_matrix.block(1,0).vmult(schur_rhs, tmp);
  schur_rhs -= system_rhs.block(1);

  // Build outer solver and compute pressure
  SchurComplement<typename InnerPreconditioner<dim>::type>
    schur_complement(system_matrix, A_inverse);
  
  SolverControl control(solution.block(1).size(), 1e-6*schur_rhs.l2_norm());
  SolverCG<Vector<double>> solver(control);

  // The inverse matrix of mass_pressure is a good preconditioner.
  SparseILU<double> preconditioner;
  preconditioner.initialize(preconditioner_matrix.block(1,1),
                            SparseILU<double>::AdditionalData());
  InverseMatrix<SparseMatrix<double>, SparseILU<double>> mass_p_inverse(
    preconditioner_matrix.block(1,1), preconditioner);

  solver.solve(schur_complement, solution.block(1), 
               schur_rhs, mass_p_inverse);
  constraints.distribute(solution);

  std::cout << "  " << control.last_step()
            << " outer CG Schur complement iterations for pressure"
            << std::endl;
  
  // Compute velocities from pressure
  system_matrix.block(0,1).vmult(tmp, solution.block(1));
  tmp *= -1.;
  tmp += system_rhs.block(0);
  A_inverse.vmult(solution.block(0), tmp);
  constraints.distribute(solution);
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


int main(int argc, const char *argv[]){
  if(argc != 3)
  {
    std::cerr << "Param error! Please run with command" << std::endl;
    std::cerr << "./stokes k N" << std::endl;
    std::cerr << "where k is the degree of FE, and N is the finest grid size."
              << std::endl;
    return -1;
  }
  StokesProblem<2> stokes(std::stoi(argv[1]),  // degree of FE
                          std::stoi(argv[2])); // grid size
  stokes.run();
  return 0;
}