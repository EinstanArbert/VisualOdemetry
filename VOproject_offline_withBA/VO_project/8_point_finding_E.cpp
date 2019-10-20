
#include <opencv2/opencv.hpp>

int run8Point(const CvMat* _m1, const CvMat* _m2, CvMat* _fmatrix)
{
	double a[9 * 9], w[9], v[9 * 9];
	CvMat W = cvMat(1, 9, CV_64F, w);
	CvMat V = cvMat(9, 9, CV_64F, v);
	CvMat A = cvMat(9, 9, CV_64F, a);
	CvMat U, F0, TF;
	CvPoint2D64f m0c = { 0, 0 }, m1c = { 0, 0 };
	double t, scale0 = 0, scale1 = 0;
	const CvPoint2D64f* m1 = (const CvPoint2D64f*)_m1->data.ptr;
	const CvPoint2D64f* m2 = (const CvPoint2D64f*)_m2->data.ptr;
	double* fmatrix = _fmatrix->data.db;
	int i, j, k, count = _m1->cols*_m1->rows;
	// compute centers and average distances for each of the two point sets
	for (i = 0; i < count; i++)
	{
		double x = m1[i].x, y = m1[i].y;
		m0c.x += x; m0c.y += y;
		x = m2[i].x, y = m2[i].y;
		m1c.x += x; m1c.y += y;
	}
	// calculate the normalizing transformations for each of the point sets:
	// after the transformation each set will have the mass center at the coordinate origin
	// and the average distance from the origin will be ~sqrt(2).
	t = 1. / count;
	m0c.x *= t; m0c.y *= t;
	m1c.x *= t; m1c.y *= t;
	for (i = 0; i < count; i++)
	{
		double x = m1[i].x - m0c.x, y = m1[i].y - m0c.y;
		scale0 += sqrt(x*x + y*y);
		x = fabs(m2[i].x - m1c.x), y = fabs(m2[i].y - m1c.y);
		scale1 += sqrt(x*x + y*y);
	}
	scale0 *= t;
	scale1 *= t;
	if (scale0 < FLT_EPSILON || scale1 < FLT_EPSILON)
		return 0;
	scale0 = sqrt(2.) / scale0;
	scale1 = sqrt(2.) / scale1;

	cvZero(&A);
	// form a linear system Ax=0: for each selected pair of points m1 & m2,
	// the row of A(=a) represents the coefficients of equation: (m2, 1)'*F*(m1, 1) = 0
	// to save computation time, we compute (At*A) instead of A and then solve (At*A)x=0. 
	for (i = 0; i < count; i++)
	{
		double x0 = (m1[i].x - m0c.x)*scale0;
		double y0 = (m1[i].y - m0c.y)*scale0;
		double x1 = (m2[i].x - m1c.x)*scale1;
		double y1 = (m2[i].y - m1c.y)*scale1;
		double r[9] = { x1*x0, x1*y0, x1, y1*x0, y1*y0, y1, x0, y0, 1 };
		for (j = 0; j < 9; j++)
			for (k = 0; k < 9; k++)
				a[j * 9 + k] += r[j] * r[k];
	}
	cvSVD(&A, &W, 0, &V, CV_SVD_MODIFY_A + CV_SVD_V_T);
	for (i = 0; i < 8; i++)
	{
		if (fabs(w[i]) < DBL_EPSILON)
			break;
	}
	if (i < 7)
		return 0;
	F0 = cvMat(3, 3, CV_64F, v + 9 * 8); // take the last column of v as a solution of Af = 0
	// make F0 singular (of rank 2) by decomposing it with SVD,
	// zeroing the last diagonal element of W and then composing the matrices back.
	// use v as a temporary storage for different 3x3 matrices
	W = U = V = TF = F0;
	W.data.db = v;
	U.data.db = v + 9;
	V.data.db = v + 18;
	TF.data.db = v + 27;
	cvSVD(&F0, &W, &U, &V, CV_SVD_MODIFY_A + CV_SVD_U_T + CV_SVD_V_T);
	W.data.db[8] = 0.;
	// F0 <- U*diag([W(1), W(2), 0])*V'
	cvGEMM(&U, &W, 1., 0, 0., &TF, CV_GEMM_A_T);
	cvGEMM(&TF, &V, 1., 0, 0., &F0, 0/*CV_GEMM_B_T*/);
	// apply the transformation that is inverse
	// to what we used to normalize the point coordinates
	{
		double tt0[] = { scale0, 0, -scale0*m0c.x, 0, scale0, -scale0*m0c.y, 0, 0, 1 };
		double tt1[] = { scale1, 0, -scale1*m1c.x, 0, scale1, -scale1*m1c.y, 0, 0, 1 };
		CvMat T0, T1;
		T0 = T1 = F0;
		T0.data.db = tt0;
		T1.data.db = tt1;
		// F0 <- T1'*F0*T0
		cvGEMM(&T1, &F0, 1., 0, 0., &TF, CV_GEMM_A_T);
		F0.data.db = fmatrix;
		cvGEMM(&TF, &T0, 1., 0, 0., &F0, 0);
		// make F(3,3) = 1
		if (fabs(F0.data.db[8]) > FLT_EPSILON)
			cvScale(&F0, &F0, 1. / F0.data.db[8]);
	}
	return 1;
}