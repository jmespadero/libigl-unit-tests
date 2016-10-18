#include <test_common.h>
#include <iostream>
#include <igl/cotmatrix.h>
#include <igl/cotmatrix_entries.h>

TEST(cotmatrix, simple)
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

  //1. Check cotmatrix_entries

  Eigen::MatrixXd C1;
  igl::cotmatrix_entries(V,F,C1);
  ASSERT_EQ(F.rows(), C1.rows());
  ASSERT_EQ(3, C1.cols());
  //All angles in unit cube measure 45 or 90 degrees
  //Their (half)cotangent must value 0.5 or 0.0
  for(int f = 0;f<C1.rows();f++)
  {
    for(int v = 0;v<3;v++)
        if (C1(f,v) > 0.1)
           ASSERT_EQ(0.5, C1(f,v));
        else
           ASSERT_EQ(0.0, C1(f,v));
    //All cotangents sum 1.0 for those triangles
    ASSERT_EQ(1.0, C1.row(f).sum());
  }

  //Check the equilateral triangles
  Eigen::MatrixXd C2;
  igl::cotmatrix_entries(V,F_equi,C2);
  ASSERT_EQ(F_equi.rows(), C2.rows());
  ASSERT_EQ(3, C2.cols());
  for(int f = 0;f<C2.rows();f++)
  {
    //Their (half)cotangent must value 0.5 / tan(M_PI / 3.0)
    for(int v = 0;v<3;v++)
       ASSERT_NEAR(0.5 / tan(M_PI / 3.0), C2(f,v), epsilon);
  }

  //Scale the cube to have huge sides
  Eigen::MatrixXd V_huge = V * 1.0e8;
  igl::cotmatrix_entries(V_huge,F,C1);
  ASSERT_EQ(F.rows(), C1.rows());
  ASSERT_EQ(3, C1.cols());
  //All angles still measure 45 or 90 degrees
  //Their (half)cotangent must value 0.5 or 0.0
  for(int f = 0;f<C1.rows();f++)
  {
    for(int v = 0;v<3;v++)
        if (C1(f,v) > 0.1)
           ASSERT_EQ(0.5, C1(f,v));
        else
           ASSERT_EQ(0.0, C1(f,v));
    //All cotangents sum 1.0 for those triangles
    ASSERT_EQ(1.0, C1.row(f).sum());
  }

  //Check the equilateral triangles of huge sides
  igl::cotmatrix_entries(V_huge,F_equi,C2);
  ASSERT_EQ(F_equi.rows(), C2.rows());
  ASSERT_EQ(3, C2.cols());
  for(int f = 0;f<C2.rows();f++)
  {
    //Their (half)cotangent must value 0.5 / tan(M_PI / 3.0)
    for(int v = 0;v<3;v++)
       ASSERT_NEAR(0.5 / tan(M_PI / 3.0), C2(f,v), epsilon);
  }

  //Scale the cube to have tiny sides
  Eigen::MatrixXd V_tiny = V * 1.0e-8;
  igl::cotmatrix_entries(V_tiny,F,C1);
  ASSERT_EQ(F.rows(), C1.rows());
  ASSERT_EQ(3, C1.cols());
  //All angles still measure 45 or 90 degrees
  //Their (half)cotangent must value 0.5 or 0.0
  for(int f = 0;f<C1.rows();f++)
  {
    for(int v = 0;v<3;v++)
        if (C1(f,v) > 0.1)
           ASSERT_NEAR(0.5, C1(f,v), epsilon);
        else
           ASSERT_EQ(0.0, C1(f,v));
    //All cotangents sum 1.0 for those triangles
    ASSERT_NEAR(1.0, C1.row(f).sum(), epsilon);
  }

  //Check the equilateral triangles of tiny sides
  igl::cotmatrix_entries(V_tiny,F_equi,C2);
  ASSERT_EQ(F_equi.rows(), C2.rows());
  ASSERT_EQ(3, C2.cols());
  for(int f = 0;f<C2.rows();f++)
  {
    //Their (half)cotangent must value 0.5 / tan(M_PI / 3.0)
    for(int v = 0;v<3;v++)
       ASSERT_NEAR(0.5 / tan(M_PI / 3.0), C2(f,v), epsilon);
  }

  //2. Check cotmatrix (Laplacian)
  //The laplacian for the cube is quite singular.
  //Each edge in a diagonal has two opposite angles of 90, with cotangent 0.0 each
  //Each edge in a side has two opposite angle of 45, with cotangen 0.5 each
  //So the cotangent matrix always are (0+0) or (0.5+0.5)
  Eigen::SparseMatrix<double> L1;
  igl::cotmatrix(V,F,L1);
  ASSERT_EQ(V.rows(), L1.rows());
  ASSERT_EQ(V.rows(), L1.cols());
  for(int f = 0;f<L1.rows();f++)
  {
    ASSERT_EQ(-3.0, L1.coeff(f,f));
    ASSERT_EQ(0.0, L1.row(f).sum());
    ASSERT_EQ(0.0, L1.col(f).sum());
  }

  //Same for tiny cube. we need to use a tolerance this time...
  igl::cotmatrix(V_tiny,F,L1);
  ASSERT_EQ(V.rows(), L1.rows());
  ASSERT_EQ(V.rows(), L1.cols());
  for(int f = 0;f<L1.rows();f++)
  {
    ASSERT_NEAR(-3.0, L1.coeff(f,f), epsilon);
    ASSERT_NEAR(0.0, L1.row(f).sum(), epsilon);
    ASSERT_NEAR(0.0, L1.col(f).sum(), epsilon);
  }

  //Check the regular tetrahedron of side sqrt(2)
  igl::cotmatrix(V,F_equi,L1);

  ASSERT_EQ(V.rows(), L1.rows());
  ASSERT_EQ(V.rows(), L1.cols());
  for(int f = 0;f<L1.rows();f++)
  {
    //Check the diagonal. Only can value 0.0 for unused vertex or -3 / tan(60)
    if (L1.coeff(f,f) < -0.1)
        ASSERT_NEAR(-3 / tan(M_PI / 3.0), L1.coeff(f,f), epsilon);
    else
        ASSERT_NEAR(0.0, L1.coeff(f,f), epsilon);
    ASSERT_EQ(0.0, L1.row(f).sum());
    ASSERT_EQ(0.0, L1.col(f).sum());
  }

  //Check the tiny regular tetrahedron
  igl::cotmatrix(V_tiny,F_equi,L1);
  //std::cout << L1 << std::endl;

  ASSERT_EQ(V.rows(), L1.rows());
  ASSERT_EQ(V.rows(), L1.cols());
  for(int f = 0;f<L1.rows();f++)
  {
    //Check the diagonal. Only can value 0.0 for unused vertex or -3 / tan(60)
    if (L1.coeff(f,f) < -0.1)
        ASSERT_NEAR(-3 / tan(M_PI / 3.0), L1.coeff(f,f), epsilon);
    else
        ASSERT_NEAR(0.0, L1.coeff(f,f), epsilon);
    ASSERT_NEAR(0.0, L1.row(f).sum(), epsilon);
    ASSERT_NEAR(0.0, L1.col(f).sum(), epsilon);
  }

}
