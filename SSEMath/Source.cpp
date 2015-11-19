#include<immintrin.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <iostream>
#include <fstream>

/*-----------------------------------------------------------Reference - 3D Math Primer Book by Fletcher Dunn and Ian Parberry --------------------------------------------------------------------------*/
/*-----------------------------------------------------------Reference - Game Engine Architecture 2 - Gregory -------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------Reference - http://www.tommesani.com/SSE.html ----------------------------------------------------------------------------------------------*/

using namespace std;

// Shuffle Parameters
#define SHUFFLE_PARAM(x,y,z,w) \
	((x) |(y) <<2 | ((z) <<4) | ((w)<<6))
#define _mm_replicate_x_ps(v) \
	_mm_shuffle_ps((v),(v),SHUFFLE_PARAM(0,0,0,0))
#define _mm_replicate_y_ps(v) \
	_mm_shuffle_ps((v),(v),SHUFFLE_PARAM(1,1,1,1))
#define _mm_replicate_z_ps(v) \
	_mm_shuffle_ps((v),(v),SHUFFLE_PARAM(2,2,2,2))
#define _mm_replicate_w_ps(v) \
	_mm_shuffle_ps((v),(v),SHUFFLE_PARAM(3,3,3,3))

#define __mm_madd_ps(a,b,c) \
	_mm_add_ps(_mm_mul_ps((a),(b)),(c))

// Matrix 4x4 and Matrix 4x4 multiplication
void M4x4_SSE(__m128 *A,__m128 *B ,__m128 *MulResult)
{

	for (int i = 0; i < 4; i++){
		MulResult[i] = _mm_mul_ps(_mm_replicate_x_ps(A[i]), B[0]);
		MulResult[i] = __mm_madd_ps(_mm_replicate_y_ps(A[i]), B[1], MulResult[i]);
		MulResult[i] = __mm_madd_ps(_mm_replicate_z_ps(A[i]), B[2], MulResult[i]);
		MulResult[i] = __mm_madd_ps(_mm_replicate_w_ps(A[i]), B[3], MulResult[i]);
	}
} 


__declspec(align(16)) float a[4] = { 0, 0, 0, 0 };

// Matrix 4x4 and 4x1 Multiplication
void M4x4V4X1_SSE(__m128 *A, __m128 &v, __m128 *MulResult){
	__m128 temp1 = _mm_load_ps(a);
	__declspec(align(16)) __m128 temp;

	for (int i = 0; i < 4; i++){
		MulResult[i] = _mm_mul_ps((A[i]), v);
		temp = _mm_hadd_ps(MulResult[i], temp1);
		temp = _mm_hadd_ps(temp, temp1);
		MulResult[i] = temp;
	}

}

// Calculate Bezier Curve
void BezierCurve(__m128 *Nodes, __m128 *M, __m128 *time, __m128 *bezResult1, __m128 *bezResult2, float *MulResult){
		
	// Multiplication of the Points Matrix and the Bezier Matrix
		M4x4_SSE(Nodes, M, bezResult1);
		/*ofstream myfile;
		myfile.open("data.csv");
		myfile << "X" << "Y" << endl;*/

		for (float i = 0; i <= 1; i += 0.01){
			*time = _mm_set_ps(i*i*i, i*i, i, 1);
			// Final Multiplication with the Time Vector
			M4x4V4X1_SSE(bezResult1, *time, bezResult2);
			_MM_TRANSPOSE4_PS(bezResult2[0], bezResult2[1], bezResult2[2], bezResult2[3]);
			_mm_storeu_ps(MulResult, *bezResult2);
			//myfile << MulResult[0] << "," << MulResult[1]<<endl;
			cout << "X: " << MulResult[0] << " Y:" << MulResult[1] << " Z: " << MulResult[2] << " W: " << MulResult[3] << endl;
			cout << "\n";
		}

		//myfile.close();
}

// Normal matrix multiplication without SSE
static float temp[4][4] = { 0 };
void MatrixMultiplication(float Matrix1[4][4], float Matrix2[4][4]){
	
	for (int i = 0; i<4; ++i)
		for (int j = 0; j<4; ++j)
			for (int k = 0; k<4; ++k)
			{
				temp[i][j] += Matrix1[i][k] * Matrix2[k][j];
			}
}
// Normal matrix multiplication without SSE
static float Temp23[1][4] = { 0 };
void MatrixMultiplication2(float Matrix1[4][4], float Matrix2[4][1]){

	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; j++)
			Temp23[0][i] += (Matrix1[i][j] * Matrix2[j][0]);
			
	}
}
__m128 mulVectorMatrixFinal(__m128 *v, __m128 *Matrix){
	__m128 Result;
	Result = _mm_mul_ps(_mm_replicate_x_ps(*v), Matrix[0]);
	Result = __mm_madd_ps(_mm_replicate_y_ps(*v), Matrix[4], Result);
	Result = __mm_madd_ps(_mm_replicate_z_ps(*v), Matrix[8], Result);
	Result = __mm_madd_ps(_mm_replicate_w_ps(*v), Matrix[12], Result);

	return Result;
}
void NormalMult(){

	__declspec(align(16)) float NodeFR[4][4] =
	{ 1, 1, 0, 0,
	2, 4, 0, 0,
	5, 4, 0, 0,
	8, 1, 0, 0 };

	__declspec(align(16)) float BezR[4][4] =
	{ 1, 0, 0, 0,
	-3, 3, 0, 0,
	3, -6, 3, 0,
	-1, 3, -3, 1 };
	__declspec(align(16)) static float Final[4][4];

	// Normal matrix multiplication without SSE
	MatrixMultiplication(NodeFR, BezR);
	for (float i = 0; i < 1; i += 0.01f){
		float time[4][1];

		time[0][0] = 1;
		time[1][0] = i;
		time[2][0] = i*i;
		time[3][0] = i*i*i;
		MatrixMultiplication2(temp, time);
	}
}
__declspec(align(16)) float MultiplicationResult[16];
__declspec(align(16)) __m128 bezResult1[4];
__declspec(align(16)) __m128 bezResult2[4];
__declspec(align(16)) __m128 b[4];
__declspec(align(16)) __m128 time[1];
__declspec(align(16)) float points[16];
__declspec(align(16)) __m128 Nodes[4];


// Multiplication of Matrices and vectors using SSE
void SSEMult(){
	// Always align the floats and __m128 using __declspec

	// Note : RIGHT MOST COLUMN IS THE FIRST COORDINATE in the ORDER X-Y-Z-W

	Nodes[0] = _mm_set_ps(8, 5, 2, 1.0f);
	Nodes[1] = _mm_set_ps(1, 4, 4, 1.0f);
	Nodes[2] = _mm_set_ps(0, 0, 0, 0.0f);
	Nodes[3] = _mm_set_ps(0, 0, 0, 0.0f);

	// Note : Bezier Matrix from the Cubic Bezier Curve Matrix Formula
	

	b[0] = _mm_set_ps(-1, 3, -3, 1);
	b[1] = _mm_set_ps(3, -6, 3, 0);
	b[2] = _mm_set_ps(-3, 3, 0, 0);
	b[3] = _mm_set_ps(1, 0, 0, 0);


	BezierCurve(Nodes, b, time, bezResult1, bezResult2, points);

}
int main(){

	// SSE multiplication for Bezier Curve
	SSEMult();

	// Normal Multiplication for Bezier Curve
	NormalMult();

	system("Pause");
}