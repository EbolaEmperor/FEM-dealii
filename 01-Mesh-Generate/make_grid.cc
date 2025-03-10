#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <iostream>
#include <fstream>

using namespace dealii;

void make_grid(Triangulation<3> &triangulation)
{
  // level=1 for the coarset grid. Set larger level for finer grid.
  const unsigned level = 3;

  // the radius of the little ball.
  const double innerR = 0.5;

  SphericalManifold<3> boundary(Point<3>(5, 5, 5));
  Triangulation<3> middle, right, tmp, tmp2;

  GridGenerator::subdivided_hyper_rectangle(
      right,
      std::vector<unsigned int>({4U<<level, 2U<<level, 2U<<level}),
      Point<3>(10, 0, 0),
      Point<3>(30, 10, 10),
      false);
  GridGenerator::hyper_shell(middle, Point<3>(5.0, 5.0, 5.0), innerR, 5.0*sqrt(3), 6, true);
  middle.reset_all_manifolds();
  for (Triangulation<3>::cell_iterator cell = middle.begin();
       cell != middle.end(); ++cell)
    for (unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f)
    {
      bool is_inner_face = true;
      for (unsigned int l = 0; l < GeometryInfo<3>::lines_per_face; ++l){
        bool is_inner_line = true;
        for (unsigned int v = 0; v < 2; ++v)
        {
          Point<3> &vertex = cell->face(f)->line(l)->vertex(v);
          if (std::abs(vertex.distance(Point<3>(5, 5, 5)) - innerR) > 1e-6)
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
  middle.set_manifold(1, boundary);
  middle.refine_global(level+1);

  GridGenerator::flatten_triangulation(middle, tmp2);
  GridGenerator::merge_triangulations(tmp2, right, triangulation);

  // Set boundary index.
  // 0 : left-face (x=0)
  // 1 : right-face (x=30)
  // 2 : front-face (y=0)
  // 3 : back-face (y=10)
  // 4 : bottom-face (z=0)
  // 5 : top-face (z=10)
  // 6 : face on the ball
  for (Triangulation<3>::active_cell_iterator cell = triangulation.begin();
       cell != triangulation.end();
       ++cell)
  {
    for (unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f)
    {
      if (cell->face(f)->at_boundary())
      {
        if (std::abs(cell->face(f)->center()[0]) < 1e-12)
        {
          cell->face(f)->set_all_boundary_ids(0);
        }
        else if (std::abs(cell->face(f)->center()[0] - 30) < 1e-12)
        {
          cell->face(f)->set_all_boundary_ids(1);
        }
        else if (std::abs(cell->face(f)->center()[1]) < 1e-12)
        {
          cell->face(f)->set_all_boundary_ids(2);
        }
        else if (std::abs(cell->face(f)->center()[1] - 10) < 1e-12)
        {
          cell->face(f)->set_all_boundary_ids(3);
        }
        else if (std::abs(cell->face(f)->center()[2]) < 1e-12)
        {
          cell->face(f)->set_all_boundary_ids(4);
        }
        else if (std::abs(cell->face(f)->center()[2] - 10) < 1e-12)
        {
          cell->face(f)->set_all_boundary_ids(5);
        }
        else
        {
          cell->face(f)->set_all_boundary_ids(6);
        }
      }
    }
  }
}

template <int dim>
void output_grid(const Triangulation<dim> &triangulation)
{
  std::ofstream out("our_grid.vtk");
  GridOut grid_out;
  grid_out.write_vtk(triangulation, out);
}

int main()
{
  Triangulation<3> triangulation;
  make_grid(triangulation);
  output_grid(triangulation);
}

