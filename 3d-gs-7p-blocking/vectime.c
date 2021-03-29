#include "defines.h"

void vectime(double *B, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb)
{

	double(*A)[NY + 2 * YSTART][NZ + 2 * ZSTART] = (double(*)[NY + 2 * YSTART][NZ + 2 * ZSTART]) B;

	int tt, xx, yy, zz;
	int t, x, y, z;
	int xbeg, ybeg, zbeg;
	int myxbeg, myybeg, myzbeg;
	int myxb, myyb, myzb;
	int wave;

	const int xblocknum = myceil(NX + T - 1, xb);
	const int yblocknum = myceil(NY + T - 1, yb);
	const int zblocknum = myceil(NZ + T - 1, zb);

#ifdef scalar_ratio
	long long cnt = 0;
#endif

	for (wave = 0; wave < myceil(T, tb) + myceil(NX + T - 1, xb) - 1 + myceil(NY + T - 1, yb) - 1 + myceil(NZ + T - 1, zb) - 1; wave++)
	{

		#pragma omp parallel for private(tt, xx, yy, zz, t, x, y, z, xbeg, ybeg, zbeg, myxbeg, myybeg, myzbeg, myxb, myyb, myzb) collapse(3) schedule(dynamic, 1)

		for (xx = 0; xx < xblocknum; xx++)
		{
			for (yy = 0; yy < yblocknum; yy++)
			{
				for (zz = 0; zz < zblocknum; zz++)
				{

					__m256d vcenter;
					__m256d vx_minus_1, vx_plus_1;
					__m256d vy_minus_1, vy_plus_1;
					__m256d vz_minus_1, vz_plus_1;
					__m256d vmiddlein, vmiddleout;
					SET_COFF;
					double tmp[4];

					xbeg = XSTART + xx * xb - (wave - xx - yy - zz) * tb;
					ybeg = YSTART + yy * yb - (wave - xx - yy - zz) * tb;
					zbeg = ZSTART + zz * zb - (wave - xx - yy - zz) * tb;

					for (tt = max(0, (wave - xx - yy - zz) * tb); tt <= min(T, (wave - xx - yy - zz + 1) * tb) - VECLEN; tt += VECLEN, xbeg -= VECLEN, ybeg -= VECLEN, zbeg -= VECLEN)
					{
						if(xbeg - VECLEN + 1 < XSTART){
							myxbeg 	= 	XSTART + VECLEN - 1;
							myxb	=	xb - (myxbeg - xbeg);
						} else {
							myxbeg	=	xbeg;
							myxb	=	xb;
						}
						if (xbeg + xb >= XSTART + NX){	
							myxb -= xbeg + xb - (XSTART + NX);
						}

						if(ybeg - VECLEN + 1 < YSTART){
							myybeg 	= 	YSTART + VECLEN - 1;
							myyb	=	yb - (myybeg - ybeg);
						} else {
							myybeg	=	ybeg;
							myyb	=	yb;
						}
						if (ybeg + yb >= YSTART + NY){	
							myyb -= ybeg + yb - (YSTART + NY);
						}

						if(zbeg - VECLEN + 1 < ZSTART){
							myzbeg 	= 	ZSTART + VECLEN - 1;
							myzb	=	zb - (myzbeg - zbeg);
						} else {
							myzbeg	=	zbeg;
							myzb	=	zb;
						}
						if (zbeg + zb >= ZSTART + NZ){	
							myzb -= zbeg + zb - (ZSTART + NZ);
						}

						if ( 	myyb <= myyb_threshold \
							||	myzb <= myzb_threshold \
							||	myxb <= myxb_threshold){
							for (t = tt; t < tt + VECLEN; t++)
							{
								for (x = max(XSTART, xbeg - (t - tt)); x < min(NX + XSTART, xbeg - (t - tt) + xb); x++)
								{
									for (y = max(YSTART, ybeg - (t - tt)); y < min(NY + YSTART, ybeg - (t - tt) + yb); y++)
									{
										for (z = max(ZSTART, zbeg - (t - tt)); z < min(NZ + ZSTART, zbeg - (t - tt) + zb); z++)
										{
											Compute_scalar(A, x, y, z);

										}
									}
								}
							}
						} else {


//	this is like a 3-D Rubik's cube
//	x indices are 
//				x1 = max(XSTART, xbeg - (t - tt))
//				x2 = myxbeg - (t - tt)
//				x3 = myxbeg + myxb - STRIDE * (t - tt) - 1
//				x4 = min(NX + XSTART, xbeg + xb - (t - tt))
//	y indices are 
//				y1 = max(YSTART, ybeg - (t - tt))
//				y2 = myybeg - (t - tt)
//				y3 = myybeg - (t - tt) + myyb
//				y4 = min(NY + YSTART, ybeg + yb - (t - tt))
//	z indices are 
//				z1 = max(ZSTART, zbeg - (t - tt))
//				z2 = myzbeg - (t - tt)
//				z3 = myzbeg - (t - tt) + myzb
//				z4 = min(NZ + ZSTART, zbeg + zb - (t - tt))
// vector code processes the inner sub-block x23 * y23 * z23
// scalar code processes other 26 sub-blocks

							for( t = tt ; t < tt + VECLEN - 1 ; t++){
								//		x12 * y14 * z13 = 6 sub-blocks
								for ( x = max(XSTART, xbeg - (t - tt)); x < myxbeg - (t - tt); x++){
									for ( y = max(YSTART, ybeg - (t - tt)); y < min(NY + YSTART, ybeg + yb - (t - tt)) ; y++){
										for (z = max(ZSTART, zbeg - (t - tt)); z < myzbeg - (t - tt) + myzb; z++)
										{
											Compute_scalar(A, x, y, z);
										}
									}
								}
								//		x24 * y12 * z13 = 4 sub-blocks
								for(x = myxbeg - (t - tt); x < min(NX + XSTART, xbeg + xb - (t - tt)); x++){
									for ( y = max(YSTART, ybeg - (t - tt)); y < myybeg - (t - tt); y++){									
										for (z = max(ZSTART, zbeg - (t - tt)); z < myzbeg - (t - tt) + myzb; z++)
										{
											Compute_scalar(A, x, y, z);
										}
									}
								}
								//		x24 * y24 * z12 = 4 sub-blocks
								for(x = myxbeg - (t - tt); x < min(NX + XSTART, xbeg + xb - (t - tt)); x++){
									for ( y = myybeg - (t - tt); y < min(NY + YSTART, ybeg + yb - (t - tt)); y++){									
										for (z = max(ZSTART, zbeg - (t - tt)); z < myzbeg - (t - tt); z++)
										{
											Compute_scalar(A, x, y, z);
										}
									}
								}
							}


							//			 * <--> xbeg
							//		4|32100000|
							//		4|43322110|
							//		to enter the vector compute code, xb should be at least VECLEN
							//		xbeg - VECLEN + 1 >= XSTART
							//		xb > VECLEN

							for (t = tt; t < tt + VECLEN; t++)
							{
								for (x = myxbeg - (t - tt); x < myxbeg + VECLEN - STRIDE * (t - tt); x++)
								{
									for (y = myybeg - (t - tt); y < myybeg - (t - tt) + myyb; y++)
									{
										for (z = myzbeg - (t - tt); z < myzbeg - (t - tt) + myzb; z++)
										{
											Compute_scalar(A, x, y, z);
										}
									}
								}
							}

							//			 * <--> xbeg
							//		4|43322110|0
							//		 |#*#*#*#*|0
							for (x = myxbeg + VECLEN - 1; x < myxbeg + VECLEN - 1 + STRIDE; x++)
							{
								for (y = myybeg - VECLEN + 1; y < myybeg - VECLEN + 1 + myyb; y++)
								{
									for (z = myzbeg - VECLEN + 1; z <= myzbeg - VECLEN + 1 + myzb - VECLEN; z += VECLEN)
									{
										vcenter = _mm256_loadu_pd(&A[x - STRIDE * 0][y + 3][z + 3]);
										vx_plus_1 = _mm256_loadu_pd(&A[x - STRIDE * 1][y + 2][z + 2]);
										vy_plus_1 = _mm256_loadu_pd(&A[x - STRIDE * 2][y + 1][z + 1]);
										vz_plus_1 = _mm256_loadu_pd(&A[x - STRIDE * 3][y + 0][z + 0]);
										transpose(vcenter, vx_plus_1, vy_plus_1, vz_plus_1, vmiddlein, vmiddleout);
										_mm256_storeu_pd(&A[x - STRIDE * 0][y + 3][z + 3], vcenter);
										_mm256_storeu_pd(&A[x - STRIDE * 1][y + 2][z + 2], vx_plus_1);
										_mm256_storeu_pd(&A[x - STRIDE * 2][y + 1][z + 1], vy_plus_1);
										_mm256_storeu_pd(&A[x - STRIDE * 3][y + 0][z + 0], vz_plus_1);
									}
								}
							}
							//			     * <--> x
							//		4|43322110|0
							//		 |0#*#*#*#|*0
							for (x = myxbeg + VECLEN; x < myxbeg + myxb - 1; x++)
							{

								y = myybeg - VECLEN + 1;
								for (z = myzbeg - VECLEN + 1; z <= myzbeg - VECLEN + 1 + myzb - VECLEN; z += VECLEN)
								{
									vcenter = _mm256_loadu_pd(&A[x + 1 - STRIDE * 0][y + 3][z + 3]);
									vx_plus_1 = _mm256_loadu_pd(&A[x + 1 - STRIDE * 1][y + 2][z + 2]);
									vy_plus_1 = _mm256_loadu_pd(&A[x + 1 - STRIDE * 2][y + 1][z + 1]);
									vz_plus_1 = _mm256_loadu_pd(&A[x + 1 - STRIDE * 3][y + 0][z + 0]);

									transpose(vcenter, vx_plus_1, vy_plus_1, vz_plus_1, vmiddlein, vmiddleout);
									_mm256_storeu_pd(&A[x + 1 - STRIDE * 0][y + 3][z + 3], vcenter);
									_mm256_storeu_pd(&A[x + 1 - STRIDE * 1][y + 2][z + 2], vx_plus_1);
									_mm256_storeu_pd(&A[x + 1 - STRIDE * 2][y + 1][z + 1], vy_plus_1);
									_mm256_storeu_pd(&A[x + 1 - STRIDE * 3][y + 0][z + 0], vz_plus_1);
								}

								for (y = myybeg - VECLEN + 1; y < myybeg - VECLEN + 1 + myyb; y++)
								{

									z = myzbeg - VECLEN + 1;

									vz_minus_1 = _mm256_set_pd(A[x - STRIDE * 3][y + 0][z - 1 + 0],
															A[x - STRIDE * 2][y + 1][z - 1 + 1],
															A[x - STRIDE][y + 2][z - 1 + 2],
															A[x][y + 3][z - 1 + 3]);

									if (z <= myzbeg - VECLEN + 1 + myzb - VECLEN)
									{
										vcenter = load_vcenter_1(x, y, z);
									}
									else
									{
										vcenter = _mm256_set_pd(A[x - STRIDE * 3][y + 0][z + 0],
																	A[x - STRIDE * 2][y + 1][z + 1],
																	A[x - STRIDE * 1][y + 2][z + 2],
																	A[x + STRIDE * 0][y + 3][z + 3]);
									}

									if (y == myybeg - VECLEN + 1 && z <= myzbeg - VECLEN + 1 + myzb - VECLEN)
									{

										vx_plus_1 = load_real_vx_plus_1_1(x, y, z);
									}
									else
									{
										vx_plus_1 = _mm256_set_pd(A[x + 1 - STRIDE * 3][y + 0][z + 0],
																A[x + 1 - STRIDE * 2][y + 1][z + 1],
																A[x + 1 - STRIDE * 1][y + 2][z + 2],
																A[x + 1 + STRIDE * 0][y + 3][z + 3]);
									}

	/*
		#*?
		#*&#*
	#*&#*
	#*&#*
	&#*
	*/
									//			     * <--> x
									//		4|43322110|0
									//		 |0#*#*#*#|*0

									for (; z <= myzbeg - VECLEN + 1 + myzb - VECLEN; z += VECLEN)
									{

										vz_plus_1 = load_vcenter_2(x, y, z);
										if (y < myybeg - VECLEN + 1 + myyb - 1)
										{
											vy_plus_1 = load_vy_plus_1_1(x, y, z);
										}
										else
										{
											vy_plus_1 = _mm256_set_pd(A[x - STRIDE * 3][y + 1][z + 0],
																	A[x - STRIDE * 2][y + 2][z + 1],
																	A[x - STRIDE][y + 3][z + 2],
																	A[x][y + 4][z + 3]);
										}
										if (y == myybeg - VECLEN + 1)
										{
											vy_minus_1 = _mm256_set_pd(A[x - STRIDE * 3][y - 1][z + 0],
																	A[x - STRIDE * 2][y + 0][z + 1],
																	A[x - STRIDE][y + 1][z + 2],
																	A[x][y + 2][z + 3]);
										}
										else
										{

											vy_minus_1 = load_vy_minus_1_1(x, y, z);
										}
										vx_minus_1 = load_vx_minus_1_1(x, y, z);
										Compute_1vector(vcenter,
														vz_minus_1, vz_plus_1,
														vx_minus_1, vx_plus_1,
														vy_minus_1, vy_plus_1);

										store_newvalue_1(vz_minus_1, x, y, z);
										vcenter = vz_plus_1;

										if (y == myybeg - VECLEN + 1)
										{

											vx_plus_1 = load_real_vx_plus_1_2(x, y, z);
										}
										else
										{
											vmiddlein = _mm256_loadu_pd(&A[x + 1][y + 3][z + 4]);
											store_vx_plus_1_1(vx_plus_1, x, y, z);
											vx_plus_1 = load_vx_plus_1_1(x, y, z);
											Input_Output_1(vmiddleout, vx_plus_1, vmiddlein);
										}

										vz_plus_1 = load_vcenter_3(x, y, z);
										if (y < myybeg - VECLEN + 1 + myyb - 1)
										{
											vy_plus_1 = load_vy_plus_1_2(x, y, z);
										}
										else
										{
											vy_plus_1 = _mm256_set_pd(A[x - STRIDE * 3][y + 1][z + 1],
																	A[x - STRIDE * 2][y + 2][z + 2],
																	A[x - STRIDE][y + 3][z + 3],
																	A[x][y + 4][z + 4]);
										}
										if (y == myybeg - VECLEN + 1)
										{
											vy_minus_1 = _mm256_set_pd(A[x - STRIDE * 3][y - 1][z + 1],
																	A[x - STRIDE * 2][y + 0][z + 2],
																	A[x - STRIDE][y + 1][z + 3],
																	A[x][y + 2][z + 4]);
										}
										else
										{
											vy_minus_1 = load_vy_minus_1_2(x, y, z);
										}
										vx_minus_1 = load_vx_minus_1_2(x, y, z);
										Compute_1vector(vcenter,
														vz_minus_1, vz_plus_1,
														vx_minus_1, vx_plus_1,
														vy_minus_1, vy_plus_1);

										store_newvalue_2(vz_minus_1, x, y, z);
										vcenter = vz_plus_1;

										if (y == myybeg - VECLEN + 1)
										{
											vx_plus_1 = load_real_vx_plus_1_3(x, y, z);
										}
										else
										{
											store_vx_plus_1_2(vx_plus_1, x, y, z);
											vx_plus_1 =load_vx_plus_1_2(x, y, z);
											Input_Output_2(vmiddleout, vx_plus_1, vmiddlein);
										}

										vz_plus_1 = load_vcenter_4(x, y, z);
										if (y < myybeg - VECLEN + 1 + myyb - 1)
										{
											vy_plus_1 = load_vy_plus_1_3(x, y, z);
										}
										else
										{
											vy_plus_1 = _mm256_set_pd(A[x - STRIDE * 3][y + 1][z + 2],
																	A[x - STRIDE * 2][y + 2][z + 3],
																	A[x - STRIDE][y + 3][z + 4],
																	A[x][y + 4][z + 5]);
										}
										if (y == myybeg - VECLEN + 1)
										{
											vy_minus_1 = _mm256_set_pd(A[x - STRIDE * 3][y - 1][z + 2],
																	A[x - STRIDE * 2][y + 0][z + 3],
																	A[x - STRIDE][y + 1][z + 4],
																	A[x][y + 2][z + 5]);
										}
										else
										{
											vy_minus_1 = load_vy_minus_1_3(x, y, z);
										}
										vx_minus_1 = load_vx_minus_1_3(x, y, z);
										Compute_1vector(vcenter,
														vz_minus_1, vz_plus_1,
														vx_minus_1, vx_plus_1,
														vy_minus_1, vy_plus_1);

										store_newvalue_3(vz_minus_1, x, y, z);
										vcenter = vz_plus_1;

										if (y == myybeg - VECLEN + 1)
										{
											vx_plus_1 = load_real_vx_plus_1_4(x, y, z);
										}
										else
										{
											store_vx_plus_1_3(vx_plus_1, x, y, z);
											vx_plus_1 = load_vx_plus_1_3(x, y, z);
											Input_Output_3(vmiddleout, vx_plus_1, vmiddlein);
										}

										if (z + VECLEN <= myzbeg - VECLEN + 1 + myzb - VECLEN)
										{
											vz_plus_1 = load_vcenter_1(x, y, z + VECLEN);
										}
										else
										{
											vz_plus_1 = _mm256_set_pd(A[x - STRIDE * 3][y + 0][z + VECLEN + 0],
																	A[x - STRIDE * 2][y + 1][z + VECLEN + 1],
																	A[x - STRIDE][y + 2][z + VECLEN + 2],
																	A[x][y + 3][z + VECLEN + 3]);
										}

										if (y < myybeg - VECLEN + 1 + myyb - 1)
										{
											vy_plus_1 = load_vy_plus_1_4(x, y, z);
										}
										else
										{
											vy_plus_1 = _mm256_set_pd(A[x - STRIDE * 3][y + 1][z + 3],
																	A[x - STRIDE * 2][y + 2][z + 4],
																	A[x - STRIDE][y + 3][z + 5],
																	A[x][y + 4][z + 6]);
										}
										if (y == myybeg - VECLEN + 1)
										{
											vy_minus_1 = _mm256_set_pd(A[x - STRIDE * 3][y - 1][z + 3],
																	A[x - STRIDE * 2][y + 0][z + 4],
																	A[x - STRIDE][y + 1][z + 5],
																	A[x][y + 2][z + 6]);
										}
										else
										{
											vy_minus_1 = load_vy_minus_1_4(x, y, z);
										}
										vx_minus_1 = load_vx_minus_1_4(x, y, z);
										Compute_1vector(vcenter,
														vz_minus_1, vz_plus_1,
														vx_minus_1, vx_plus_1,
														vy_minus_1, vy_plus_1);

										store_newvalue_4(vz_minus_1, x, y, z);


										vcenter = vz_plus_1;

										if (y == myybeg - VECLEN + 1)
										{
											if (z + VECLEN <= myzbeg - VECLEN + 1 + myzb - VECLEN){
												vx_plus_1 = load_real_vx_plus_1_1(x, y, z + VECLEN);
											} else {
												vx_plus_1 = _mm256_set_pd(	A[x + 1 - STRIDE * 3][y + 0][z + VECLEN],
																			A[x + 1 - STRIDE * 2][y + 1][z + VECLEN +1],
																			A[x + 1 - STRIDE][y + 2][z + VECLEN +2],
																			A[x + 1][y + 3][z + VECLEN +3]);
											}
										}
										else
										{
											store_vx_plus_1_4(vx_plus_1, x, y, z);
											vx_plus_1 = load_vx_plus_1_4(x, y, z);
											Input_Output_4(vmiddleout, vx_plus_1, vmiddlein);
											_mm256_storeu_pd(&A[x + 1 - STRIDE * VECLEN][y - 1][z], vmiddleout);


										}
									}
									if(y > myybeg - VECLEN + 1){
										_mm256_storeu_pd(tmp, vx_plus_1);
										A[x + 1 - STRIDE * 0][y + 3][z + 3] = tmp[0];
										A[x + 1 - STRIDE * 1][y + 2][z + 2] = tmp[1];
										A[x + 1 - STRIDE * 2][y + 1][z + 1] = tmp[2];
										A[x + 1 - STRIDE * 3][y + 0][z + 0] = tmp[3];
									}
									for (z += 1; z <= myzbeg - VECLEN + 1 + myzb; z++)
									{

										vx_minus_1 = _mm256_set_pd(A[x - 1 - STRIDE * 3][y + 0][z + 0 - 1],
																A[x - 1 - STRIDE * 2][y + 1][z + 1 - 1],
																A[x - 1 - STRIDE][y + 2][z + 2 - 1],
																A[x - 1][y + 3][z + 3 - 1]);

										vx_plus_1 = _mm256_set_pd(A[x + 1 - STRIDE * 3][y + 0][z + 0 - 1],
																A[x + 1 - STRIDE * 2][y + 1][z + 1 - 1],
																A[x + 1 - STRIDE][y + 2][z + 2 - 1],
																A[x + 1][y + 3][z + 3 - 1]);

										vy_minus_1 = _mm256_set_pd(A[x - STRIDE * 3][y + 0 - 1][z + 0 - 1],
																A[x - STRIDE * 2][y + 1 - 1][z + 1 - 1],
																A[x - STRIDE][y + 2 - 1][z + 2 - 1],
																A[x][y + 3 - 1][z + 3 - 1]);

										vy_plus_1 = _mm256_set_pd(A[x - STRIDE * 3][y + 0 + 1][z + 0 - 1],
																A[x - STRIDE * 2][y + 1 + 1][z + 1 - 1],
																A[x - STRIDE][y + 2 + 1][z + 2 - 1],
																A[x][y + 3 + 1][z + 3 - 1]);

										vz_plus_1 = _mm256_set_pd(A[x - STRIDE * 3][y + 0][z + 0],
																A[x - STRIDE * 2][y + 1][z + 1],
																A[x - STRIDE][y + 2][z + 2],
																A[x][y + 3][z + 3]);

										Compute_1vector(vcenter,
														vz_minus_1, vz_plus_1,
														vx_minus_1, vx_plus_1,
														vy_minus_1, vy_plus_1);

										_mm256_storeu_pd(tmp, vz_minus_1);

										A[x][y + 3][z + 3 - 1] = tmp[0];
										A[x - STRIDE][y + 2][z + 2 - 1] = tmp[1];
										A[x - STRIDE * 2][y + 1][z + 1 - 1] = tmp[2];
										A[x - STRIDE * 3][y + 0][z + 0 - 1] = tmp[3];

										vcenter = vz_plus_1;
									}

								}

								y = myybeg - VECLEN + 1 + myyb - 1;
								for (z = myzbeg - VECLEN + 1; z <= myzbeg - VECLEN + 1 + myzb - VECLEN; z += VECLEN)
								{

									vcenter = _mm256_loadu_pd(&A[x - 1 - STRIDE * 0][y + 3][z + 3]);
									vx_plus_1 = _mm256_loadu_pd(&A[x - 1 - STRIDE * 1][y + 2][z + 2]);
									vy_plus_1 = _mm256_loadu_pd(&A[x - 1 - STRIDE * 2][y + 1][z + 1]);
									vz_plus_1 = _mm256_loadu_pd(&A[x - 1 - STRIDE * 3][y + 0][z + 0]);

									transpose(vcenter, vx_plus_1, vy_plus_1, vz_plus_1, vmiddlein, vmiddleout);

									_mm256_storeu_pd(&A[x - 1 - STRIDE * 0][y + 3][z + 3], vcenter);
									_mm256_storeu_pd(&A[x - 1 - STRIDE * 1][y + 2][z + 2], vx_plus_1);
									_mm256_storeu_pd(&A[x - 1 - STRIDE * 2][y + 1][z + 1], vy_plus_1);
									_mm256_storeu_pd(&A[x - 1 - STRIDE * 3][y + 0][z + 0], vz_plus_1);
								}
							}

							for (x = myxbeg + myxb - 2; x < myxbeg + myxb; x++)
							{
								for (y = myybeg - VECLEN + 1; y < myybeg - VECLEN + 1 + myyb; y++)
								{
									for (z = myzbeg - VECLEN + 1; z <= myzbeg - VECLEN + 1 + myzb - VECLEN; z += VECLEN)
									{
										vcenter = _mm256_loadu_pd(&A[x - STRIDE * 0][y + 3][z + 3]);
										vx_plus_1 = _mm256_loadu_pd(&A[x - STRIDE * 1][y + 2][z + 2]);
										vy_plus_1 = _mm256_loadu_pd(&A[x - STRIDE * 2][y + 1][z + 1]);
										vz_plus_1 = _mm256_loadu_pd(&A[x - STRIDE * 3][y + 0][z + 0]);
										transpose(vcenter, vx_plus_1, vy_plus_1, vz_plus_1, vmiddlein, vmiddleout);
										_mm256_storeu_pd(&A[x - STRIDE * 0][y + 3][z + 3], vcenter);
										_mm256_storeu_pd(&A[x - STRIDE * 1][y + 2][z + 2], vx_plus_1);
										_mm256_storeu_pd(&A[x - STRIDE * 2][y + 1][z + 1], vy_plus_1);
										_mm256_storeu_pd(&A[x - STRIDE * 3][y + 0][z + 0], vz_plus_1);
									}
								}
							}

							//			 * <--> xbeg
							//		4|43322110|0
							//		 | #*#*#*#|*

							for (t = tt; t < tt + VECLEN; t++)
							{
								//	x34 * y23 * z23 = 1 sub-block
								for (x = myxbeg + myxb - STRIDE * (t - tt) - 1; x < min(NX + XSTART, xbeg + xb - (t - tt)); x++){
									for (y = myybeg - (t - tt); y < myybeg - (t - tt) + myyb; y++){
										for (z = myzbeg - (t - tt); z < myzbeg - (t - tt) + myzb; z++){
											Compute_scalar(A, x, y, z);
										}
									}
								}
								//	x24 * y34 * z23 = 2 sub-blocks
								for(x = myxbeg - (t - tt); x < min(NX + XSTART, xbeg + xb - (t - tt)); x++){
									for ( y = myybeg + myyb - (t - tt); y < min(NY + YSTART, ybeg + yb - (t - tt)); y++){									
										for (z = myzbeg - (t - tt); z < myzbeg - (t - tt) + myzb; z++)
										{
											Compute_scalar(A, x, y, z);
										}
									}
								}
								//	x14 * y14 * z34 = 9 sub-blocks
								for(x = max(XSTART, xbeg - (t - tt)); x < min(NX + XSTART, xbeg + xb - (t - tt)); x++){
									for ( y = max(YSTART, ybeg - (t - tt)); y < min(NY + YSTART, ybeg + yb - (t - tt)); y++){									
										for (z = myzbeg + myzb - (t - tt); z < min(NZ + ZSTART, zbeg + zb - (t - tt)); z++)
										{
											Compute_scalar(A, x, y, z);
										}
									}
								}
							}	
						}
					}

					// note that it should be x = max(XSTART, xbeg - (t - (wave - xx - yy) * tb))
					// and x < min(T + XSTART, xbeg - (t - (wave - xx - yy) * tb) + xb)
					// but if (wave - xx - yy) * tb < 0, then max(0, (wave - xx - yy) * tb) <= 0
					// then this block contains no computation as (wave - xx - yy + 1) * tb) <= 0
					// y is similar
					for (t = tt; t < min(T, (wave - xx - yy - zz + 1) * tb); t++, xbeg--, ybeg--, zbeg--)
					{
						for (x = max(XSTART, xbeg); x < min(NX + XSTART, xbeg + xb); x++)
						{
							for (y = max(YSTART, ybeg); y < min(NY + YSTART, ybeg + yb); y++)
							{
								for (z = max(ZSTART, zbeg); z < min(NZ + ZSTART, zbeg + zb); z++)
								{
									Compute_scalar(A, x, y, z);
								}
							}
						}
					}
				}
			}
		}
	}
#ifdef scalar_ratio
	printf("%f\n", (double) cnt /(double)((double)NX * (double) NY *(double) NZ * (double) T));
#endif
}

							/*

	4444443322110
	4444443322110
	4444443322110
	333333221100
	22222211000
	1111110000


	4444443322210
	4444443322210
	4444443322210
	333333221110
	22222211000
	1111110000

	4444443333210
	4444443333210
	4444444443210
	333333333210
	22222222210
	1111111110


	4444443322110
	4444443322110
	4444443322110
	333333221100
	22222211000
	1111110000


	4444443322110
	333333221110
	22222211110
	1111111110


	4444444443210
	4444444443210
	4444444443210
	4444444443210 -----ybeg
	333333333210
	22222222210
	1111111110
	*/