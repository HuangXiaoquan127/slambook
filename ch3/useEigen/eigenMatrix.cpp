#include <iostream>
using namespace std;
#include <ctime>

#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 50

int main( int argc, char** argv )
{
    Eigen::Matrix<float, 2, 3> matrix_23;
    Eigen::Vector3d v_3d;
    Eigen::Matrix<float, 3, 1> vd_3d;
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > matrix_dynamic;
    
    Eigen::MatrixXd matrix_x;
    
    matrix_23 << 1, 2, 3, 4, 5, 6;
    //matrix_23 << (1, 2, 3, 4, 5, 6);//Wrong form, 不加括号 
    cout << matrix_23 << endl;
    
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            cout << matrix_23(i, j) << "\t";
            //matrix_23(i, j) = 0;
            //用于引出编译错误，锁定程序真正还原的功能函数的位置，调试是索引不到的，因为此有些程序在编译时就已经被转换和替换好了；
            //查看是如何使用括号把元素取出的，其实也是重载了（）符号，使类操作起来像函数一样；
            //matrix_23(i, 2, 8) = 0;
        }
        cout << endl;
    }
    //cout << matrix_23 << endl;
    
    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;
    
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << result << endl;
    
    Eigen::Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    cout << result2 << endl;
    
    //Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;
    
    matrix_33 = Eigen::Matrix3d::Random();
    cout << matrix_33 << endl << endl;
    
    cout << matrix_33.transpose() << endl;
    cout << matrix_33.sum() << endl;
    cout << matrix_33.trace() << endl;
    cout << 10 * matrix_33 << endl;
    cout << matrix_33.inverse() << endl;
    cout << matrix_33.determinant() << endl;
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver( matrix_33.transpose() * matrix_33);
    cout << "Real sysmetric matrix = \n" << matrix_33.transpose() * matrix_33 << endl;
    cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;
    
    Eigen::Matrix< double, MATRIX_SIZE, MATRIX_SIZE > matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE );
    Eigen::Matrix< double, MATRIX_SIZE, 1 > v_Nd;
    v_Nd = Eigen::MatrixXd::Random( MATRIX_SIZE, 1 );
    
    clock_t time_stt = clock();
    
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time use in normal inverse is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time use in Qr decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    
    return 0;    
}