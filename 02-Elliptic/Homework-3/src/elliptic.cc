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

using namespace dealii;

class Solution : public Function<2>{
public:
  double value(const Point<2> & p, const unsigned int component = 0) const{
    return cos(M_PI*p[0]) * cos(M_PI*p[1]);
  };

  virtual Tensor<1, 2>
  gradient(const Point<2> & p, const unsigned int component = 0) const{
    Tensor<1, 2> grad;
    grad[0] = -M_PI * sin(M_PI*p[0]) * cos(M_PI*p[1]);
    grad[1] = -M_PI * cos(M_PI*p[0]) * sin(M_PI*p[1]);
    return grad;
  };
};

double f(const Point<2> &p){
  return (2*M_PI*M_PI+1) * cos(M_PI*p[0]) * cos(M_PI*p[1]);
}

class Elliptic{
public:
  Elliptic(const int &level);
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  int level;
  Triangulation<2> triangulation;
  FE_SimplexP<2>   fe;
  DoFHandler<2>    dof_handler;
  std::map<types::global_dof_index, Point<2>> dof_location_map;
 
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
 
  Vector<double> solution;
  Vector<double> error;
  Vector<double> system_rhs;
};


Elliptic::Elliptic(const int &level): level(level), fe(1), dof_handler(triangulation){}


void Elliptic::make_grid(){
  Triangulation<2> tmp;
  GridGenerator::hyper_cube(tmp);
  tmp.refine_global(2+level);
  GridGenerator::convert_hypercube_to_simplex_mesh(tmp, triangulation);
  std::ofstream mesh_file("mesh.vtk");
  GridOut().write_vtk(triangulation, mesh_file);
}


void Elliptic::setup_system(){
  dof_handler.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dof_handler);

  dof_location_map = DoFTools::map_dofs_to_support_points(MappingFE<2>(fe), dof_handler);
  std::ofstream dof_location_file("dof-locations.gnuplot");
  DoFTools::write_gnuplot_dof_support_point_info(dof_location_file,
                                                 dof_location_map);

  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                  dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);

  std::ofstream out("sparsity-pattern.svg");
  sparsity_pattern.print_svg(out);

  system_matrix.reinit(sparsity_pattern);
 
  solution.reinit(dof_handler.n_dofs());
  error.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

 
void Elliptic::assemble_system()
{
  QGaussSimplex<2> quadrature_formula(fe.degree + 1);
  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values | update_quadrature_points);
 
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
 
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
 
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
 
      cell_matrix = 0;
      cell_rhs    = 0;
      const auto &quadrature_points =  fe_values.get_quadrature_points();
 
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) *   // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) +   // grad phi_j(x_q)
                 fe_values.shape_value(i, q_index) *  // phi_i(x_q)
                 fe_values.shape_value(j, q_index)    // phi_j(x_q)
                ) * fe_values.JxW(q_index);           // dx
 
          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            f(quadrature_points[q_index]) *     // f(x_q)
                            fe_values.JxW(q_index));            // dx
        }
      cell->get_dof_indices(local_dof_indices);
 
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));
 
      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
}


void Elliptic::solve()
{
  SolverControl            solver_control(1000, 1e-6 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

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
                                    VectorTools::L1_norm);
  const double L1_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L1_norm);
  std::cout << "L1-norm error: " << L1_error << std::endl;

  difference_per_cell.reinit(triangulation.n_active_cells());
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
  std::cout << "L2-norm error: " << L2_error << std::endl;

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
  std::cout << "Linfty-norm error: " << Linf_error << std::endl;

  difference_per_cell.reinit(triangulation.n_active_cells());
  VectorTools::integrate_difference(MappingFE<2>(fe),
                                    dof_handler,
                                    solution,
                                    Solution(),
                                    difference_per_cell,
                                    QGaussSimplex<2>(fe.degree + 1),
                                    VectorTools::H1_seminorm);
  const double H1_semi_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::H1_seminorm);
  std::cout << "H1-seminorm error: " << H1_semi_error << std::endl;

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
  std::cout << "H1-norm error: " << H1_error << std::endl;
}


void Elliptic::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}

 
int main(int argc, char* argv[])
{
  if(argc != 2){
    std::cerr << "Param error!" << std::endl;
    return -1;
  }
  Elliptic solver(std::stoi(argv[1]));
  solver.run();
  return 0;
}