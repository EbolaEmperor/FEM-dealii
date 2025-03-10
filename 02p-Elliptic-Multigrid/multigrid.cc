#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
 
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
 
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
 
#include <deal.II/dofs/dof_tools.h>
 
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
 
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
 
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


template <int dim>
class Elliptic
{
public:
  Elliptic();
  void run();

private:
  template <class Iterator>
  void cell_worker(const Iterator    &cell,
                   ScratchData<dim>  &scratch_data,
                   CopyData          &copy_data);

  void setup_system();
  void assemble_system();
  void assemble_multigrid();
  void solve();
  void make_grid();
  void refine_grid();
  void output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;

  AffineConstraints<double> constraints;

  MGLevelObject<SparsityPattern> mg_sparsity_patterns;
  MGLevelObject<SparsityPattern> mg_interface_sparsity_patterns;

  MGLevelObject<SparseMatrix<double>> mg_matrices;
  MGLevelObject<SparseMatrix<double>> mg_interface_matrices;
  MGConstrainedDoFs                   mg_constrained_dofs;
};


/*************************************************
 * Solve the PDE : -a \Delta u = 1  in domain
 *                           u = 0  on boundary
 * "a" at point p is given by coefficient(p).
**************************************************/

template <int dim>
double coefficient(const Point<dim> &p)
{
  if (p.square() < 1.25 * 1.25)
    return 20;
  else
    return 1;
}

template <int dim>
Elliptic<dim>::Elliptic()
  : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
  , fe(1)
  , dof_handler(triangulation)
{}

template <int dim>
void Elliptic<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << " (by level: ";
  for (unsigned int level = 0; level < triangulation.n_levels(); ++level)
    std::cout << dof_handler.n_dofs(level)
              << (level == triangulation.n_levels() - 1 ? ")" : ", ");
  std::cout << std::endl;

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  { // destroy dsp after use
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);
  }
  system_matrix.reinit(sparsity_pattern);

  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(dof_handler);
  std::set<types::boundary_id> dirichlet_boundary_ids = {0};
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                     dirichlet_boundary_ids);

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
}


template <int dim>
template <class Iterator>
void Elliptic<dim>::cell_worker(const Iterator     &cell,
                                ScratchData<dim>   &scratch_data,
                                CopyData           &copy_data)
{
  FEValues<dim> &fe_values = scratch_data.fe_values;
  fe_values.reinit(cell);

  const unsigned dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();

  copy_data.reinit(cell, dofs_per_cell);
  auto q_points = fe_values.get_quadrature_points();

  for(const unsigned q : fe_values.quadrature_point_indices())
  {
    const double weight = fe_values.JxW(q);
    const double coef = coefficient(q_points[q]);

    for(unsigned i = 0; i < dofs_per_cell; ++i)
    {
      for(unsigned j = 0; j < dofs_per_cell; ++j)
      {
        copy_data.cell_matrix(i,j) += coef * 
                                      fe_values.shape_grad(i, q) *
                                      fe_values.shape_grad(j, q) *
                                      weight;
      }
      copy_data.cell_rhs(i) += fe_values.shape_value(i,q) * weight;
    }
  }
}


template <int dim>
void Elliptic<dim>::assemble_system()
{
  MappingQ<dim> mapping(fe.degree);

  auto cell_worker = 
   [&] (const typename DoFHandler<dim>::active_cell_iterator &cell,
        ScratchData<dim>                                     &scratch_data,
        CopyData                                             &copy_data) {
      this->cell_worker(cell, scratch_data, copy_data);
    };
  
  auto copier = [&](const CopyData &cd){
    this->constraints.distribute_local_to_global(cd.cell_matrix,
                                                 cd.cell_rhs,
                                                 cd.local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
  };

  const unsigned n_gauss_points = fe.degree+1;

  ScratchData<dim> scratch_data(mapping,
                                fe,
                                n_gauss_points,
                                update_values | update_gradients | 
                                  update_JxW_values | update_quadrature_points);
  
  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        CopyData(),
                        MeshWorker::assemble_own_cells);
}


template <int dim>
void Elliptic<dim>::assemble_multigrid()
{
  MappingQ<dim> mapping(fe.degree);
  const unsigned n_levels = triangulation.n_levels();

  std::vector<AffineConstraints<double>> boundary_constraints(n_levels);
  for (unsigned int level = 0; level < n_levels; ++level)
    {
      const IndexSet dofset =
        DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
      boundary_constraints[level].reinit(dofset);
      boundary_constraints[level].add_lines(
        mg_constrained_dofs.get_refinement_edge_indices(level));
      boundary_constraints[level].add_lines(
        mg_constrained_dofs.get_boundary_indices(level));
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
                                  update_JxW_values | update_quadrature_points);

  std::cerr << "start mesh loop" << std::endl;

  MeshWorker::mesh_loop(dof_handler.begin_mg(),
                        dof_handler.end_mg(),
                        cell_worker,
                        copier,
                        scratch_data,
                        CopyData(),
                        MeshWorker::assemble_own_cells);
}


template <int dim>
void Elliptic<dim>::solve()
{
  MGTransferPrebuilt<Vector<double>> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  FullMatrix<double> coarse_matrix;
  coarse_matrix.copy_from(mg_matrices[0]);
  MGCoarseGridHouseholder<double, Vector<double>> coarse_grid_solver;
  coarse_grid_solver.initialize(coarse_matrix);

  using Smoother = PreconditionSOR<SparseMatrix<double>>;
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

  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  solution = 0;
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
  constraints.distribute(solution);
}


template <int dim>
void Elliptic<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim-1>(fe.degree+2),
                                     {},                                 // Here for Neumann conditions.
                                     solution,
                                     estimated_error_per_cell);
  
  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.4,                       // Refine 40% cells
                                                  0.1);                      // Coarsen 10% cells
  // We could do other things here, but nothing is needed in this program.
  triangulation.execute_coarsening_and_refinement();
}


template <int dim>
void Elliptic<dim>::output_results(const unsigned int cycle) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
  data_out.write_vtu(output);
}


template <int dim>
void Elliptic<dim>::make_grid(){
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
void Elliptic<dim>::run()
{
  Assert(dim == 2, ExcInternalError());
  make_grid();

  for (unsigned int cycle = 0; cycle <= 10; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        triangulation.refine_global(2);
      else
        refine_grid();

      std::cout << "   Number of active cells: "  //
                << triangulation.n_active_cells() //
                << std::endl                      //
                << "   Total number of cells: "   //
                << triangulation.n_cells()        //
                << std::endl;

      setup_system();
      assemble_system();
      assemble_multigrid();
      solve();
      output_results(cycle);
    }
}

int main()
{
  Elliptic<2> laplace_problem_2d;
  laplace_problem_2d.run();
  return 0;
}
