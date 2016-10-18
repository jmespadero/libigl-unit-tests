#include <test_common.h>
#include <igl/edge_lengths_squared.h>
#include <igl/edge_lengths.h>
#include <iostream>

TEST(edge_lengths, simple)
{
    //The allowed error for this test
    const double epsilon = 1e-15;

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  //This is a cube of dimensions 1.0x1.0x1.0
  test_common::load_mesh("cube.obj", V, F);

  //Prepare another mesh with triangles along side diagonals of the cube
  //These triangles are form a regular tetrahedron of side sqrt(2)
  Eigen::MatrixXi F_equi(4,3);
  F_equi << 4,6,1,
            6,4,3,
            4,1,3,
            1,6,3;

  //1. Check edge_lengths_squared

  double scale = 1.0;
  double side_sq = 1.0; //squared lenght of a side
  double diag_sq = 2.0; //squared lenght of a diagonal
  Eigen::MatrixXd L_sq;
  igl::edge_lengths_squared(V,F,L_sq);
  ASSERT_EQ(F.rows(), L_sq.rows());
  ASSERT_EQ(3, L_sq.cols());
  //All edges in unit cube measure 1.0 or sqrt(2) in diagonals
  for(int f = 0;f<L_sq.rows();f++)
  {
    //All edge_lengths_squared must be exactly "side_sq" or "diag_sq"
    for(int e = 0;e<3;e++)
        if (L_sq(f,e) > 1.1)
           ASSERT_EQ(diag_sq, L_sq(f,e));
        else
           ASSERT_EQ(side_sq, L_sq(f,e));
    //All sides sum exactly side_sq + side_sq + diag_sq
    ASSERT_EQ(L_sq.row(f).sum(), side_sq + side_sq + diag_sq);
  }

  //Check the equilateral triangles
  igl::edge_lengths_squared(V,F_equi,L_sq);
  ASSERT_EQ(F_equi.rows(), L_sq.rows());
  ASSERT_EQ(3, L_sq.cols());
  //All edges measure sqrt(2)
  for(int f = 0;f<L_sq.rows();f++)
  {
      //All edge_lengths_squared must be exactly "diag_sq"
    for(int e = 0;e<3;e++)
       ASSERT_EQ(2.0, L_sq(f,e));
  }


  //Scale the cube to have huge sides
  scale = 1.0e8;
  side_sq = scale * scale;  //squared lenght of a side
  diag_sq = 2.0 * side_sq;  //squared lenght of a diagonal
  Eigen::MatrixXd V_huge = V * scale;  
  igl::edge_lengths_squared(V_huge,F,L_sq);
  ASSERT_EQ(F.rows(), L_sq.rows());
  ASSERT_EQ(3, L_sq.cols());
  for(int f = 0;f<L_sq.rows();f++)
  {
    //All edge_lengths_squared must be exactly side_sq or diag_sq
    for(int e = 0;e<3;e++)
        if (L_sq(f,e) > 1.1*side_sq)
           ASSERT_EQ(diag_sq, L_sq(f,e));
        else
           ASSERT_EQ(side_sq, L_sq(f,e));
    //All sides sum exactly side_sq + side_sq + diag_sq
    ASSERT_EQ(L_sq.row(f).sum(), side_sq + side_sq + diag_sq);
  }
 
  //Check the equilateral triangles
  igl::edge_lengths_squared(V_huge,F_equi,L_sq);
  ASSERT_EQ(F_equi.rows(), L_sq.rows());
  ASSERT_EQ(3, L_sq.cols());
  //All edges measure sqrt(2)
  for(int f = 0;f<L_sq.rows();f++)
  {
    //All edge_lengths_squared must be exactly "diag_sq"
    for(int e = 0;e<3;e++)
       ASSERT_EQ(diag_sq, L_sq(f,e));
  }

  //Scale the cube to have tiny sides
  scale = 1.0e-8;
  side_sq = scale * scale;  //squared lenght of a side
  diag_sq = 2.0 * side_sq;  //squared lenght of a diagonal
  Eigen::MatrixXd V_tiny = V * scale;
  igl::edge_lengths_squared(V_tiny,F,L_sq);
  ASSERT_EQ(F.rows(), L_sq.rows());
  ASSERT_EQ(3, L_sq.cols());
  for(int f = 0;f<L_sq.rows();f++)
  {
    //All edge_lengths_squared must be exactly side_sq or diag_sq
    for(int e = 0;e<3;e++)
        if (L_sq(f,e) > 1.1*side_sq)
           ASSERT_EQ(diag_sq, L_sq(f,e));
        else
           ASSERT_EQ(side_sq, L_sq(f,e));
    //All sides sum exactly side_sq + side_sq + diag_sq
    ASSERT_EQ(L_sq.row(f).sum(), side_sq + side_sq + diag_sq);
  }

  //Check the equilateral triangles
  igl::edge_lengths_squared(V_tiny,F_equi,L_sq);
  ASSERT_EQ(F_equi.rows(), L_sq.rows());
  ASSERT_EQ(3, L_sq.cols());
  //All edges measure sqrt(2)
  for(int f = 0;f<L_sq.rows();f++)
  {
    //All edge_lengths_squared must be exactly "diag_sq"
    for(int e = 0;e<3;e++)
       ASSERT_EQ(diag_sq, L_sq(f,e));
  }

  //Invalidate L_sq
  L_sq.resize(0,0);

  //2. Check edge_lengths
  double side = 1.0;       //lenght of a side
  double diag = sqrt(2.0); //lenght of a diagonal
  Eigen::MatrixXd L;
  igl::edge_lengths(V,F,L);
  ASSERT_EQ(F.rows(), L.rows());
  ASSERT_EQ(3, L.cols());
  //All edges in unit cube measure 1.0 or sqrt(2) in diagonals
  for(int f = 0;f<L.rows();f++)
  {
    //All edge_lengths_squared must be exactly "side" or "diag"
    for(int e = 0;e<3;e++)
        if (L(f,e) > 1.1*side)
           ASSERT_EQ(diag, L(f,e));
        else
           ASSERT_EQ(side, L(f,e));
    //All sides sum exactly side + side + diag
    ASSERT_EQ(L.row(f).sum(), side + side + diag);
  }

  //Check the cube of huge sides
  scale = 1.0e8;
  side = scale;       //lenght of a side
  diag = scale*sqrt(2.0); //lenght of a diagonal
  igl::edge_lengths(V_huge,F,L);
  ASSERT_EQ(F.rows(), L.rows());
  ASSERT_EQ(3, L.cols());
  for(int f = 0;f<L.rows();f++)
  {
    //All edge_lengths_squared must be exactly "side" or "diag"
    for(int e = 0;e<3;e++)
        if (L(f,e) > 1.1*side)
           ASSERT_EQ(diag, L(f,e));
        else
           ASSERT_EQ(side, L(f,e));
    //All sides sum exactly side + side + diag
    ASSERT_EQ(L.row(f).sum(), side + side + diag);
  }

  //Check the cube of tiny sides
  scale = 1.0e-8;
  side = scale;       //lenght of a side
  diag = scale*sqrt(2.0); //lenght of a diagonal
  igl::edge_lengths(V_tiny,F,L);
  ASSERT_EQ(F.rows(), L.rows());
  ASSERT_EQ(3, L.cols());
  for(int f = 0;f<L.rows();f++)
  {
    //All edge_lengths_squared must be exactly "side" or "diag"
    for(int e = 0;e<3;e++)
        if (L(f,e) > 1.1*side)
           ASSERT_EQ(diag, L(f,e));
        else
           ASSERT_EQ(side, L(f,e));
    //All sides sum exactly side + side + diag
    ASSERT_EQ(L.row(f).sum(), side + side + diag);
  }

}
