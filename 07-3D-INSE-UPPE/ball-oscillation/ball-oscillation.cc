//----------------------------------------------------------
// This code is for ball-oscillation test
// 
// Velocity-pressure decomposition method: UPPE
//   (See the formula (19) of Jianguo Liu (2010) )
// Time discretization: IMEX-trapezoidal (2nd order)
// Space discretization: Q2 element (2nd order in H1)
//----------------------------------------------------------

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
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <fstream>
#include <iostream>

using namespace dealii;

const unsigned Raynolds = 1000;
const double diffusion_coefficient = 1.0 / Raynolds;

//--------------------------Data Structures for MG--------------------------


template <int dim>
struct ScratchData
{
  ScratchData(const Mapping<dim>         &mapping,
              const FiniteElement<dim>   &fe,
              const unsigned              q_dgree,
              const UpdateFlags           upd_flags)
    : fe_values(mapping, fe, QGauss<dim>(q_dgree), upd_flags)
  {}

  ScratchData(const ScratchData<dim> &scratch_data)
    : fe_values(scratch_data.fe_values.get_mapping(),
                scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags())
  {}

  FEValues<dim> fe_values;
};


struct CopyData
{
  unsigned                                 level;
  FullMatrix<double>                       cell_matrix;
  Vector<double>                           cell_rhs;
  std::vector<types::global_dof_index>     local_dof_indices;

  template <class Iterator>
  void reinit(const Iterator &cell, unsigned dofs_per_cell)
  {
    cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    cell_rhs.reinit(dofs_per_cell);

    local_dof_indices.resize(dofs_per_cell);
    cell->get_active_or_mg_dof_indices(local_dof_indices);
    level = cell->level();
  }
};


//----------------------------Boundary Values----------------------------

template <int dim>
class InflowBoundaryTerm : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};

template <int dim>
double InflowBoundaryTerm<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  (void)component;
  static const double Um = 24.0;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return Um * p[1] * (1.0-p[1]) * p[2] * (1.0-p[2]) 
         * sin(this->get_time() * M_PI/4);
}


template <int dim>
class InflowBoundaryTermDt : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};

template <int dim>
double InflowBoundaryTermDt<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  (void)component;
  static const double Um = 24.0;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return Um * p[1] * (1.0-p[1]) * p[2] * (1.0-p[2]) 
         * cos(this->get_time() * M_PI/4) * M_PI/4;
}


//-------------------------------Solver Class---------------------------------


template <int dim>
class INSE{
public:
  INSE(const int &, const double &);
  void run();

private:
  template <class Iterator>
  void cell_worker(const Iterator    &cell,
                   ScratchData<dim>  &scratch_data,
                   CopyData          &copy_data);
  void assemble_multigrid();

  void make_mesh();
  void setup_system();
  void setup_convection(const Vector<double>& u1, 
                        const Vector<double>& u2,
                        const Vector<double>& u3);
  void setup_grad_pressure();
  void make_constraints_u1(const double &t);
  void update_pressure(const Vector<double>& u1,
                       const Vector<double>& u2,
                       const Vector<double>& u3);
  void solve_time_step(Vector<double>& solution, const bool ispressure = false);
  void compute_vortricity();
  void output_result(const bool force_output = false);

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints_u1;
  AffineConstraints<double> constraints_u2;
  AffineConstraints<double> constraints_u3;
  AffineConstraints<double> constraints_pressure;

  SparsityPattern      sparsity_pattern;
  SparsityPattern      sparsity_pattern_pressure;

  SparseMatrix<double> laplace_matrix_pressure;
  SparseMatrix<double> system_matrix_pressure;

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> laplace_matrix;
  SparseMatrix<double> system_matrix;

  MGLevelObject<SparsityPattern> mg_sparsity_patterns;
  MGLevelObject<SparsityPattern> mg_interface_sparsity_patterns;

  MGLevelObject<SparseMatrix<double>> mg_matrices;
  MGLevelObject<SparseMatrix<double>> mg_interface_matrices;
  MGConstrainedDoFs                   mg_constrained_dofs;

  Vector<double> convection_u1;
  Vector<double> convection_u2;
  Vector<double> convection_u3;
  Vector<double> grad_pressure_u1;
  Vector<double> grad_pressure_u2;
  Vector<double> grad_pressure_u3;

  Vector<double> pressure;
  Vector<double> vortricity;
  Vector<double> solution_u1;
  Vector<double> solution_u2;
  Vector<double> solution_u3;
  Vector<double> prev_solution_u1;
  Vector<double> prev_solution_u2;
  Vector<double> prev_solution_u3;
  Vector<double> system_rhs;

  int      level;
  double   end_time;
  double   time;
  double   time_step;
  unsigned timestep_number;
};


//---------------------------------Solver Implemention-----------------------------------


template <int dim>
INSE<dim>::INSE
  (const int &N, const double &T)
  : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
  , fe(2)
  , dof_handler(triangulation)
  , level(N)
  , end_time(T)
  , time_step(4e-4)
{}


template <int dim>
void INSE<dim>::make_mesh(){
  const double innerR = 0.2;
  const Point<3> center(0.5, 0.5, 0.5);

  SphericalManifold<3> boundary(center);
  GridGenerator::hyper_shell(triangulation, center, 
                             innerR, 0.5*sqrt(3), 6, true);
  triangulation.reset_all_manifolds();
  for (Triangulation<3>::cell_iterator cell = triangulation.begin();
       cell != triangulation.end(); ++cell)
    for (unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f)
    {
      bool is_inner_face = true;
      for (unsigned int l = 0; l < GeometryInfo<3>::lines_per_face; ++l){
        bool is_inner_line = true;
        for (unsigned int v = 0; v < 2; ++v)
        {
          Point<3> &vertex = cell->face(f)->line(l)->vertex(v);
          if (std::abs(vertex.distance(center) - innerR) > 1e-6)
          {
            is_inner_line = false;
            is_inner_face = false;
            break;
          }
        }
        if (is_inner_line)
          cell->face(f)->line(l)->set_manifold_id(1);
      }
      if (is_inner_face)
        cell->face(f)->set_manifold_id(1);
    }
  triangulation.set_manifold(1, boundary);
  triangulation.refine_global(level);

  // Set boundary index.
  // 1 : left-face (x=0)
  // 2 : right-face (x=1)
  // 0 : other faces
  for (Triangulation<3>::active_cell_iterator cell = triangulation.begin();
       cell != triangulation.end();
       ++cell)
    for (unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
      {
        if (std::abs(cell->face(f)->center()[0]) < 1e-12)
          cell->face(f)->set_all_boundary_ids(1);
        else if (std::abs(cell->face(f)->center()[0] - 1.0) < 1e-12)
          cell->face(f)->set_all_boundary_ids(2);
        else
          cell->face(f)->set_all_boundary_ids(0);
      }
  std::cerr << "make_mesh done. cell: " << triangulation.n_active_cells() << std::endl;
}

template <int dim>
void INSE<dim>::compute_vortricity()
{
  Assert(dim == 3, ExcNotImplemented());
  system_matrix.copy_from(mass_matrix);
  system_rhs.reinit(solution_u1.size());

  QGauss<dim> quadrature_formula(fe.degree+1);
  FEValues<dim> fe_val_vel(fe,
                            quadrature_formula,
                            update_gradients | update_JxW_values |
                              update_values);
  const unsigned dpc = fe.n_dofs_per_cell(),
                 nqp = quadrature_formula.size();

  std::vector<types::global_dof_index> ldi(dpc);
  Vector<double>                       loc_rot(dpc);

  std::vector<Tensor<1, dim>> grad_u2(nqp), grad_u3(nqp);
  vortricity = 0.;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_val_vel.reinit(cell);
      cell->get_dof_indices(ldi);
      fe_val_vel.get_function_gradients(solution_u2, grad_u2);
      fe_val_vel.get_function_gradients(solution_u3, grad_u3);
      loc_rot = 0.;
      for (unsigned int q = 0; q < nqp; ++q)
        for (unsigned int i = 0; i < dpc; ++i)
          loc_rot(i) += (grad_u3[q][1] - grad_u2[q][2]) * //
                        fe_val_vel.shape_value(i, q) *    //
                        fe_val_vel.JxW(q);

      for (unsigned int i = 0; i < dpc; ++i)
        system_rhs(ldi[i]) += loc_rot(i);
    }

  solve_time_step(vortricity);
}


template <int dim>
void INSE<dim>::output_result(const bool force_output)
{
  if(!force_output && (timestep_number % 50)) return;
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_u1, "solution_u1");
  data_out.add_data_vector(solution_u2, "solution_u2");
  data_out.add_data_vector(solution_u3, "solution_u3");
  data_out.add_data_vector(pressure, "pressure");
  compute_vortricity();
  data_out.add_data_vector(vortricity, "vortricity_yz");

  data_out.build_patches();
  std::ofstream output("solution/solution-" + std::to_string(timestep_number) + ".vtu");
  data_out.write_vtu(output);
}


template <int dim>
void INSE<dim>::solve_time_step(Vector<double>& solution, const bool ispressure)
{
  SolverControl            solver_control(2000, ispressure ? 1e-8 : 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  if(!ispressure)
  {
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
  } else {
    MGTransferPrebuilt<Vector<double>> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    SolverControl coarse_solver_control(5000, 1e-9, false, false);
    SolverCG<Vector<double>> coarse_solver(coarse_solver_control);
    PreconditionIdentity id;
    MGCoarseGridIterativeSolver<Vector<double>,
                                SolverCG<Vector<double>>,
                                SparseMatrix<double>,
                                PreconditionIdentity>
        coarse_grid_solver(coarse_solver, mg_matrices[0], id);

    using Smoother = PreconditionChebyshev<SparseMatrix<double>, Vector<double>>;
    mg::SmootherRelaxation<Smoother, Vector<double>> mg_smoother;
    mg_smoother.initialize(mg_matrices);
    mg_smoother.set_steps(2);
    mg_smoother.set_symmetric(true);

    mg::Matrix<Vector<double>> mg_matrix(mg_matrices);
    mg::Matrix<Vector<double>> mg_interface_up(mg_interface_matrices);
    mg::Matrix<Vector<double>> mg_interface_down(mg_interface_matrices);

    Multigrid<Vector<double>> mg(
      mg_matrix, coarse_grid_solver, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface_down, mg_interface_up);

    PreconditionMG< dim, Vector<double>, MGTransferPrebuilt<Vector<double>> >
      preconditioner(dof_handler, mg, mg_transfer);
    
    solver.solve(system_matrix_pressure, solution, system_rhs, preconditioner);

    const double mean_value = VectorTools::compute_mean_value(
      dof_handler, QGauss<dim>(fe.degree + 1), solution, 0);
    solution.add(-mean_value);
  }

  std::cout << "   " << solver_control.last_step()
            << " CG iterations." << std::endl;
}

template <int dim>
void INSE<dim>::make_constraints_u1(const double &t)
{
  constraints_u1.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints_u1);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u1);
  InflowBoundaryTerm<dim> boundary;
  boundary.set_time(t);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           boundary,
                                           constraints_u1);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           2,
                                           boundary,
                                           constraints_u1);
  constraints_u1.close();
}


template <int dim>
void INSE<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dof_handler);

  std::cout << std::endl
            << "===========================================" << std::endl
            << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

  make_constraints_u1(time);

  constraints_u2.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints_u2);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u2);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u2);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           2,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u2);
  constraints_u2.close();

  constraints_u3.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints_u3);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u3);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u3);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           2,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u3);
  constraints_u3.close();

  constraints_pressure.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints_pressure);
  constraints_pressure.close();

  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_u1);
    sparsity_pattern.copy_from(dsp);
  }
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_pressure);
    sparsity_pattern_pressure.copy_from(dsp);
  }

  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  laplace_matrix_pressure.reinit(sparsity_pattern_pressure);
  system_matrix_pressure.reinit(sparsity_pattern_pressure);

  MatrixCreator::create_mass_matrix(dof_handler,
                                    QGauss<dim>(fe.degree+1),
                                    mass_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<dim>(fe.degree+1),
                                       laplace_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<dim>(fe.degree+1),
                                       laplace_matrix_pressure);
  laplace_matrix *= diffusion_coefficient;
  
  vortricity.reinit(dof_handler.n_dofs());
  pressure.reinit(dof_handler.n_dofs());
  solution_u1.reinit(dof_handler.n_dofs());
  solution_u2.reinit(dof_handler.n_dofs());
  solution_u3.reinit(dof_handler.n_dofs());
  prev_solution_u1.reinit(dof_handler.n_dofs());
  prev_solution_u2.reinit(dof_handler.n_dofs());
  prev_solution_u3.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  assemble_multigrid();
}


template <int dim>
template <class Iterator>
void INSE<dim>::cell_worker(const Iterator     &cell,
                            ScratchData<dim>   &scratch_data,
                            CopyData           &copy_data)
{
  FEValues<dim> &fe_values = scratch_data.fe_values;
  fe_values.reinit(cell);

  const unsigned dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();
  copy_data.reinit(cell, dofs_per_cell);

  for(const unsigned q : fe_values.quadrature_point_indices())
  {
    const double weight = fe_values.JxW(q);
    for(unsigned i = 0; i < dofs_per_cell; ++i)
      for(unsigned j = 0; j < dofs_per_cell; ++j)
      {
        copy_data.cell_matrix(i,j) += fe_values.shape_grad(i, q) *
                                      fe_values.shape_grad(j, q) *
                                      weight;
      }
    }
}


template <int dim>
void INSE<dim>::assemble_multigrid()
{
  dof_handler.distribute_mg_dofs();
  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(dof_handler);
  
  const unsigned n_levels = triangulation.n_levels();

  mg_interface_matrices.resize(0, n_levels-1);
  mg_matrices.resize(0, n_levels-1);
  mg_sparsity_patterns.resize(0, n_levels-1);
  mg_interface_sparsity_patterns.resize(0, n_levels-1);

  for(unsigned level = 0; level < n_levels; ++level)
  {
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs(level));
      MGTools::make_sparsity_pattern(dof_handler, dsp, level);
      mg_sparsity_patterns[level].copy_from(dsp);
      mg_matrices[level].reinit(mg_sparsity_patterns[level]);
    }
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs(level));
      MGTools::make_interface_sparsity_pattern(dof_handler,
                                               mg_constrained_dofs,
                                               dsp,
                                               level);
      mg_interface_sparsity_patterns[level].copy_from(dsp);
      mg_interface_matrices[level].reinit(mg_interface_sparsity_patterns[level]);
    }
  }

  MappingQ<dim> mapping(fe.degree);

  std::vector<AffineConstraints<double>> boundary_constraints(n_levels);
  for (unsigned int level = 0; level < n_levels; ++level)
    {
      const IndexSet dofset =
        DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
      boundary_constraints[level].reinit(dofset);
      boundary_constraints[level].add_lines(
        mg_constrained_dofs.get_refinement_edge_indices(level));
      boundary_constraints[level].close();
    }

  auto cell_worker = 
   [&] (const typename DoFHandler<dim>::level_cell_iterator  &cell,
        ScratchData<dim>                                     &scratch_data,
        CopyData                                             &copy_data) {
      this->cell_worker(cell, scratch_data, copy_data);
    };

  auto copier = [&](const CopyData &cd){
    boundary_constraints[cd.level].distribute_local_to_global(
                                               cd.cell_matrix,
                                               cd.local_dof_indices,
                                               mg_matrices[cd.level]);
    const unsigned dofs_per_cell = cd.local_dof_indices.size();
    
    for(unsigned i = 0; i < dofs_per_cell; ++i)
      for(unsigned j = 0; j < dofs_per_cell; ++j)
        if(mg_constrained_dofs.is_interface_matrix_entry(
              cd.level, cd.local_dof_indices[i], cd.local_dof_indices[j]))
        {
          mg_interface_matrices[cd.level].add(cd.local_dof_indices[i],
                                              cd.local_dof_indices[j],
                                              cd.cell_matrix(i, j));
        }
  };

  const unsigned n_gauss_points = fe.degree+1;

  ScratchData<dim> scratch_data(mapping,
                                fe,
                                n_gauss_points,
                                update_values | update_gradients | 
                                  update_JxW_values);

  MeshWorker::mesh_loop(dof_handler.begin_mg(),
                        dof_handler.end_mg(),
                        cell_worker,
                        copier,
                        scratch_data,
                        CopyData(),
                        MeshWorker::assemble_own_cells);
}


template <int dim>
void INSE<dim>::setup_convection
      (const Vector<double>& u1, 
       const Vector<double>& u2,
       const Vector<double>& u3)
{
  convection_u1.reinit(solution_u1.size());
  convection_u2.reinit(solution_u1.size());
  convection_u3.reinit(solution_u1.size());

  const unsigned dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double> cell_convection_u1(dofs_per_cell);
  Vector<double> cell_convection_u2(dofs_per_cell);
  Vector<double> cell_convection_u3(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  QGauss<dim> quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_JxW_values);
  std::vector<double> u1_q_point(quadrature_formula.size());
  std::vector<double> u2_q_point(quadrature_formula.size());
  std::vector<double> u3_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u1_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u2_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u3_q_point(quadrature_formula.size());

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_convection_u1 = 0.;
      cell_convection_u2 = 0.;
      cell_convection_u3 = 0.;

      fe_values.reinit(cell);
      fe_values.get_function_values(u1, u1_q_point);
      fe_values.get_function_values(u2, u2_q_point);
      fe_values.get_function_values(u3, u3_q_point);
      fe_values.get_function_gradients(u1, grad_u1_q_point);
      fe_values.get_function_gradients(u2, grad_u2_q_point);
      fe_values.get_function_gradients(u3, grad_u3_q_point);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        double weight = fe_values.JxW(q_index);

        Tensor<1,dim> ugu;

        ugu[0] =   grad_u1_q_point[q_index][0] * u1_q_point[q_index]
                 + grad_u1_q_point[q_index][1] * u2_q_point[q_index]
                 + grad_u1_q_point[q_index][2] * u3_q_point[q_index];

        ugu[1] =   grad_u2_q_point[q_index][0] * u1_q_point[q_index]
                 + grad_u2_q_point[q_index][1] * u2_q_point[q_index]
                 + grad_u2_q_point[q_index][2] * u3_q_point[q_index];

        ugu[2] =   grad_u3_q_point[q_index][0] * u1_q_point[q_index]
                 + grad_u3_q_point[q_index][1] * u2_q_point[q_index]
                 + grad_u3_q_point[q_index][2] * u3_q_point[q_index];

        for (const unsigned int i : fe_values.dof_indices())
        {
          cell_convection_u1(i) += weight * fe_values.shape_value(i, q_index) * ugu[0];
          cell_convection_u2(i) += weight * fe_values.shape_value(i, q_index) * ugu[1];
          cell_convection_u3(i) += weight * fe_values.shape_value(i, q_index) * ugu[2];
        }
      }

      // distribute to global
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
      {
        convection_u1[local_dof_indices[i]] += cell_convection_u1[i];
        convection_u2[local_dof_indices[i]] += cell_convection_u2[i];
        convection_u3[local_dof_indices[i]] += cell_convection_u3[i];
      }
    }
}


template <int dim>
void INSE<dim>::setup_grad_pressure()
{
  grad_pressure_u1.reinit(solution_u1.size());
  grad_pressure_u2.reinit(solution_u1.size());
  grad_pressure_u3.reinit(solution_u1.size());

  const unsigned dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double> cell_rhs_u1(dofs_per_cell);
  Vector<double> cell_rhs_u2(dofs_per_cell);
  Vector<double> cell_rhs_u3(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  QGauss<dim> quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_JxW_values);
  std::vector<Tensor<1,dim>> grad_pressure_q_point(quadrature_formula.size());

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_rhs_u1 = 0.;
      cell_rhs_u2 = 0.;
      cell_rhs_u3 = 0.;
      fe_values.reinit(cell);
      fe_values.get_function_gradients(pressure, grad_pressure_q_point);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        double weight = fe_values.JxW(q_index);
        for (const unsigned int i : fe_values.dof_indices())
        {
          double phi = fe_values.shape_value(i, q_index);
          cell_rhs_u1(i) += weight * phi * 
                            grad_pressure_q_point[q_index][0];
          cell_rhs_u2(i) += weight * phi * 
                            grad_pressure_q_point[q_index][1];
          cell_rhs_u3(i) += weight * phi * 
                            grad_pressure_q_point[q_index][2];
        }
      }

      // distribute to global
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
      {
        grad_pressure_u1[local_dof_indices[i]] += cell_rhs_u1(i);
        grad_pressure_u2[local_dof_indices[i]] += cell_rhs_u2(i);
        grad_pressure_u3[local_dof_indices[i]] += cell_rhs_u3(i);
      }
    }
}


template <int dim>
void INSE<dim>::update_pressure(
  const Vector<double> &u1,
  const Vector<double> &u2,
  const Vector<double> &u3)
{
  system_matrix_pressure.copy_from(laplace_matrix_pressure);
  system_rhs.reinit(solution_u1.size());

  const unsigned dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  QGauss<dim> quadrature_formula(fe.degree+1);
  QGauss<dim-1> quadrature_formula_face(fe.degree+1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_JxW_values | update_quadrature_points);
  const MappingQ<dim> mapping(fe.degree);
  FEFaceValues<dim> fe_face_values(mapping,
                                   fe,
                                   quadrature_formula_face,
                                   update_values | update_gradients |
                                   update_JxW_values | update_normal_vectors
                                    | update_quadrature_points);

  std::vector<double> u1_q_point(quadrature_formula.size());
  std::vector<double> u2_q_point(quadrature_formula.size());
  std::vector<double> u3_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u1_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u2_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u3_q_point(quadrature_formula.size());

  std::vector<Tensor<1,dim>> grad_u1_face_q_point(quadrature_formula_face.size());
  std::vector<Tensor<1,dim>> grad_u2_face_q_point(quadrature_formula_face.size());
  std::vector<Tensor<1,dim>> grad_u3_face_q_point(quadrature_formula_face.size());

  InflowBoundaryTermDt<dim> boundary_dt;
  boundary_dt.set_time(time);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_rhs = 0.;
      fe_values.reinit(cell);
      fe_values.get_function_values(u1, u1_q_point);
      fe_values.get_function_values(u2, u2_q_point);
      fe_values.get_function_values(u3, u3_q_point);
      fe_values.get_function_gradients(u1, grad_u1_q_point);
      fe_values.get_function_gradients(u2, grad_u2_q_point);
      fe_values.get_function_gradients(u3, grad_u3_q_point);
      auto q_points = fe_values.get_quadrature_points();

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        double weight = fe_values.JxW(q_index);

        Tensor<1,dim> ugu;

        ugu[0] = - grad_u1_q_point[q_index][0] * u1_q_point[q_index]
                 - grad_u1_q_point[q_index][1] * u2_q_point[q_index]
                 - grad_u1_q_point[q_index][2] * u3_q_point[q_index];

        ugu[1] = - grad_u2_q_point[q_index][0] * u1_q_point[q_index]
                 - grad_u2_q_point[q_index][1] * u2_q_point[q_index]
                 - grad_u2_q_point[q_index][2] * u3_q_point[q_index];

        ugu[2] = - grad_u3_q_point[q_index][0] * u1_q_point[q_index]
                 - grad_u3_q_point[q_index][1] * u2_q_point[q_index]
                 - grad_u3_q_point[q_index][2] * u3_q_point[q_index];
                 
        for (const unsigned int i : fe_values.dof_indices())
          cell_rhs(i) += weight * ugu * fe_values.shape_grad(i, q_index);
      }

      for (const auto &f : cell->face_iterators()){
        if( ! (f->at_boundary()) ) continue;
        fe_face_values.reinit(cell, f);
        fe_face_values.get_function_gradients(u1, grad_u1_face_q_point);
        fe_face_values.get_function_gradients(u2, grad_u2_face_q_point);
        fe_face_values.get_function_gradients(u3, grad_u3_face_q_point);
        auto q_face_points = fe_face_values.get_quadrature_points();
        
        for (const unsigned int q_index : fe_face_values.quadrature_point_indices())
        {
          Tensor<1,dim> normal = fe_face_values.normal_vector(q_index);
          double weight = fe_face_values.JxW(q_index);

          auto gdt = boundary_dt.value(q_face_points[q_index]);
          if(f->boundary_id() != 1 && f->boundary_id() != 2) gdt = 0;

          Tensor<1,dim> vor_u;
          vor_u[0] = grad_u3_face_q_point[q_index][1] - grad_u2_face_q_point[q_index][2];
          vor_u[1] = grad_u1_face_q_point[q_index][2] - grad_u3_face_q_point[q_index][0];
          vor_u[2] = grad_u2_face_q_point[q_index][0] - grad_u1_face_q_point[q_index][1];

          for (const unsigned int i : fe_face_values.dof_indices())
          {
            auto grad = fe_face_values.shape_grad(i, q_index);
            Tensor<1,dim> normal_grad_phi;
            normal_grad_phi[0] = normal[1]*grad[2] - normal[2]*grad[1];
            normal_grad_phi[1] = normal[2]*grad[0] - normal[0]*grad[2];
            normal_grad_phi[2] = normal[0]*grad[1] - normal[1]*grad[0];
            
            cell_rhs(i)    += diffusion_coefficient * vor_u * normal_grad_phi * weight
                            - normal[0] * gdt * fe_face_values.shape_value(i,q_index) * weight;
          }
        }
      }

      // distribute to global
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        system_rhs[local_dof_indices[i]] += cell_rhs(i);
    }
  
  constraints_pressure.condense(system_matrix_pressure, system_rhs);
  solve_time_step(pressure, true);
  constraints_pressure.distribute(pressure);
}


template <int dim>
void INSE<dim>::run(){
  Vector<double> tmp;
  Vector<double> middle_solution_u1;
  Vector<double> middle_solution_u2;
  Vector<double> middle_solution_u3;

  Vector<double> grad_pressure_u1_stage1;
  Vector<double> grad_pressure_u2_stage1;
  Vector<double> grad_pressure_u3_stage1;
  Vector<double> convection_u1_stage1;
  Vector<double> convection_u2_stage1;
  Vector<double> convection_u3_stage1;

  make_mesh();
  setup_system();

  time            = 0.0;
  timestep_number = 0;

  tmp.reinit(solution_u1.size());
  middle_solution_u1.reinit(solution_u1.size());
  middle_solution_u2.reinit(solution_u1.size());
  middle_solution_u3.reinit(solution_u1.size());
  
  VectorTools::interpolate(dof_handler,
                           Functions::ZeroFunction<dim>(),
                           prev_solution_u1);
  VectorTools::interpolate(dof_handler,
                           Functions::ZeroFunction<dim>(),
                           prev_solution_u2);
  VectorTools::interpolate(dof_handler,
                           Functions::ZeroFunction<dim>(),
                           prev_solution_u3);

  solution_u1 = prev_solution_u1;
  solution_u2 = prev_solution_u2;
  solution_u3 = prev_solution_u3;

  while(time <= end_time){
    // ------------------------------first stage-------------------------------------
    setup_convection(prev_solution_u1, prev_solution_u2, prev_solution_u3);
    update_pressure(prev_solution_u1, prev_solution_u2, prev_solution_u3);
    output_result();

    time += time_step;
    timestep_number++;
    make_constraints_u1(time);
    std::cout << std::endl << "Time step " << timestep_number << " at t=" << time << std::endl;

    setup_grad_pressure();

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(0.5*time_step, laplace_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u1);
    laplace_matrix.vmult(tmp, prev_solution_u1);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-time_step, convection_u1);
    system_rhs.add(-time_step, grad_pressure_u1);

    constraints_u1.condense(system_matrix, system_rhs);
    solve_time_step(middle_solution_u1);
    constraints_u1.distribute(middle_solution_u1);

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(0.5*time_step, laplace_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u2);
    laplace_matrix.vmult(tmp, prev_solution_u2);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-time_step, convection_u2);
    system_rhs.add(-time_step, grad_pressure_u2);
    
    constraints_u2.condense(system_matrix, system_rhs);
    solve_time_step(middle_solution_u2);
    constraints_u2.distribute(middle_solution_u2);

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(0.5*time_step, laplace_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u3);
    laplace_matrix.vmult(tmp, prev_solution_u3);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-time_step, convection_u3);
    system_rhs.add(-time_step, grad_pressure_u3);
    
    constraints_u3.condense(system_matrix, system_rhs);
    solve_time_step(middle_solution_u3);
    constraints_u3.distribute(middle_solution_u3);


    //------------------------------second stage-------------------------------------

    convection_u1_stage1 = convection_u1;
    convection_u2_stage1 = convection_u2;
    convection_u3_stage1 = convection_u3;
    grad_pressure_u1_stage1 = grad_pressure_u1;
    grad_pressure_u2_stage1 = grad_pressure_u2;
    grad_pressure_u3_stage1 = grad_pressure_u3;

    setup_convection(middle_solution_u1, middle_solution_u2, middle_solution_u3);
    update_pressure(middle_solution_u1, middle_solution_u2, middle_solution_u3);
    setup_grad_pressure();

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u1);
    laplace_matrix.vmult(tmp, prev_solution_u1);
    system_rhs.add(-0.5*time_step, tmp);
    laplace_matrix.vmult(tmp, middle_solution_u1);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-0.5*time_step, convection_u1);
    system_rhs.add(-0.5*time_step, convection_u1_stage1);
    system_rhs.add(-0.5*time_step, grad_pressure_u1);
    system_rhs.add(-0.5*time_step, grad_pressure_u1_stage1);

    constraints_u1.condense(system_matrix, system_rhs);
    solve_time_step(solution_u1);
    constraints_u1.distribute(solution_u1);

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u2);
    laplace_matrix.vmult(tmp, middle_solution_u2);
    system_rhs.add(-0.5*time_step, tmp);
    laplace_matrix.vmult(tmp, prev_solution_u2);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-0.5*time_step, convection_u2);
    system_rhs.add(-0.5*time_step, convection_u2_stage1);
    system_rhs.add(-0.5*time_step, grad_pressure_u2);
    system_rhs.add(-0.5*time_step, grad_pressure_u2_stage1);

    constraints_u2.condense(system_matrix, system_rhs);
    solve_time_step(solution_u2);
    constraints_u2.distribute(solution_u2);

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u3);
    laplace_matrix.vmult(tmp, middle_solution_u3);
    system_rhs.add(-0.5*time_step, tmp);
    laplace_matrix.vmult(tmp, prev_solution_u3);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-0.5*time_step, convection_u3);
    system_rhs.add(-0.5*time_step, convection_u3_stage1);
    system_rhs.add(-0.5*time_step, grad_pressure_u3);
    system_rhs.add(-0.5*time_step, grad_pressure_u3_stage1);

    constraints_u3.condense(system_matrix, system_rhs);
    solve_time_step(solution_u3);
    constraints_u3.distribute(solution_u3);

    prev_solution_u1 = solution_u1;
    prev_solution_u2 = solution_u2;
    prev_solution_u3 = solution_u3;
  }

  setup_convection(prev_solution_u1, prev_solution_u2, prev_solution_u3);
  update_pressure(prev_solution_u1, prev_solution_u2, prev_solution_u3);
  std::cout << std::endl;
  output_result(true);
}


int main(int argc, const char *argv[]){
  if(argc != 3){
    std::cerr << "Param error! Please run with command" << std::endl;
    std::cerr << "./vortex-ring N T" << std::endl;
    std::cerr << "where N is the level of base grid, T is end_time." << std::endl;
    return -1;
  }
  int level = std::stoi(argv[1]);
  double end_time = std::stod(argv[2]);
  INSE<3> inse(level, end_time);
  inse.run();
  return 0;
}