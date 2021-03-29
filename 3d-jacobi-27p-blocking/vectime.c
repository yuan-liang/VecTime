#include "defines.h"

void vectime(double *A, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb) {
	
    double (* B)[NX + 2 * XSTART][ NY + 2 * YSTART][ NZ + 2 * ZSTART] =  (double (*)[NX + 2 * XSTART][ NY + 2 * YSTART][ NZ + 2 * ZSTART]) A;
#ifdef scalar_ratio
	long long cnt = 0;
#endif
    tb -= tb % 4;
    int Bx = xb;
	int bx = Bx - 2 * tb * XSLOPE;
	int ix = Bx + bx;   // ix is even
	int nb0[2] = { myfloor(NX-Bx,ix), myfloor(NX-Bx,ix) + 1 };	
	int nrestpoints = NX % ix;
	int bx_first_B1 = (Bx + nrestpoints)/2;
	int bx_last_B1  = (Bx + nrestpoints) - bx_first_B1;
	int xright[2] = {bx_first_B1 + Bx + XSTART,  bx_first_B1 + (Bx - bx)/2 + XSTART};
    // printf("Bx:%d bx: %d ix:%d nb0:%d %d\nnrest:%d bxfirst:%d xright:%d %d\n",    \
            Bx,   bx,    ix,  nb0[0],nb0[1],nrestpoints,bx_first_B1,xright[0],xright[1]);
	int wave;
	int mylevel;
	int tt, xx, yy, zz, t, tv, x, y, z;
	int xmin, xmax, ybeg, zbeg, myybeg, myzbeg;
    int myyb, myzb;

	const int xblocknum = max(nb0[0], nb0[1]);
	const int yblocknum = 2 * myceil(NY + T - 1, yb);
	const int zblocknum = 2 * myceil(NZ + T - 1, zb);

    int myid = omp_get_max_threads();

	double (* AV) [(yb + 2)][(zb + 2)][VECLEN] = (double(*)[(yb + 2)][(zb + 2)][VECLEN])alloc_extra_array(sizeof(double) * (yb + 2) * (zb + 2) * VECLEN * 4 * myid);

    for (wave = 0; wave < myceil(T, tb) + 1 + myceil(NY + T - 1, yb) + myceil(NZ + T - 1, zb); wave++) {

		#pragma omp parallel for private(myid, mylevel, t, tt, tv, x, y, z, xx, yy, zz, xmin, xmax, ybeg, zbeg, myybeg, myzbeg, myyb, myzb) collapse(3) schedule(dynamic, 1)

		for (xx = 0; xx < xblocknum; xx++){
			for (yy = 0; yy < yblocknum; yy++){
				for (zz = 0; zz < zblocknum; zz++){

                    double tmp[4];

                    vec v_center_0, v_center_1, v_center_2, v_center_3;
                    __m256d v_all_d_1_0, v_all_d_1_1, v_all_d_1_2;
                    __m256d v_all_d_2_0, v_all_d_2_1, v_all_d_2_2;
                    vec in, out;
	                SET_COFF;

                    myid = omp_get_thread_num();
                    double (* BV0) [zb][VECLEN] = (double(*)[zb][VECLEN]) (AV + myid * 4 + 0);
                    double (* BV1) [zb][VECLEN] = (double(*)[zb][VECLEN]) (AV + myid * 4 + 1);
                    double (* BV2) [zb][VECLEN] = (double(*)[zb][VECLEN]) (AV + myid * 4 + 2);
                    double (* BV3) [zb][VECLEN] = (double(*)[zb][VECLEN]) (AV + myid * 4 + 3);
                    double (* Btmp [4]) [zb][VECLEN]  = {BV0, BV1, BV2, BV3};

					mylevel = (wave % 2 + yy + zz) % 2;

					if(xx < nb0[mylevel]){

						tt = -tb + (wave - yy - zz) * tb;

						ybeg = YSTART + yy * yb - (wave - yy - zz) * tb;
						zbeg = ZSTART + zz * zb - (wave - yy - zz) * tb;

						for (tv = max(tt, 0); tv <= min(tt + 2 * tb, T) - VECLEN; tv += VECLEN, ybeg -= VECLEN, zbeg -= VECLEN) {
                            
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

                            if (    xb   <= myxb_threshold \
                                ||  myyb <= myyb_threshold \
							    ||	myzb <= myzb_threshold )
                                {
							    for (t = tv; t < tv + VECLEN; t++) {
                                    xmin = (mylevel == 1 && xx == 0) ?	             XSTART : (xright[mylevel] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
						            xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel]      + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);
							    	
                                    for (x = xmin; x < xmax; x++) {
							    		for (y = max(YSTART, ybeg - (t - tv)); y < min(NY + YSTART, ybeg - (t - tv) + yb); y++) {
                                            #pragma ivdep
									        #pragma vector always
							    			for (z = max(ZSTART, zbeg - (t - tv)); z < min(NZ + ZSTART, zbeg - (t - tv) + zb); z++) {
                                                
                                                Compute_scalar(B, t, x, y, z);
							    			}
							    		}
							    	}
							    }
                            }
                            else {
                                //head
                                for (t = tv; t < tv + VECLEN - 1; t++){

						    	    xmin = (mylevel == 1 && xx == 0) ?	             XSTART : (xright[mylevel] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
						            xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel]      + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);

                                    for (x = xmin; x < xmax; x++) {
								    	for ( y = max(YSTART, ybeg - (t - tv)); y < myybeg - (t - tv); y++){									
								    		#pragma vector always
						        		    #pragma ivdep
                                            for (z = max(ZSTART, zbeg - (t - tv)); z < myzbeg - (t - tv) + myzb; z++)
								    		{
								    			Compute_scalar(B, t, x, y, z);

								    		}
								    	}
								    }
								    for (x = xmin; x < xmax; x++) {
								    	for ( y = myybeg - (t - tv); y < min(NY + YSTART, ybeg + yb - (t - tv)); y++){									
								    		#pragma vector always
						        		    #pragma ivdep
                                            for (z = max(ZSTART, zbeg - (t - tv)); z < myzbeg - (t - tv); z++)
								    		{
								    			Compute_scalar(B, t, x, y, z);

								    		}
								    	}
								    }
						        }
                                for (t = tv; t < tv + VECLEN; t++) {

						        	xmin = (mylevel == 1 && xx == 0) ? XSTART : (xright[mylevel] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
						        	xmax = xmin + STRIDE * (VECLEN - 1 - (t - tv));
						        	if (!(mylevel == 1 && xx == 0)) {
						        		xmax += ((tt + tb < tv + 1) ? 1 : -1) * (VECLEN - 1 - (t - tv));
						        	}
						        	for (x = xmin; x < xmax + 2; x++) {

						        		for (y = myybeg - (t - tv); y < myybeg - (t - tv) + myyb ; y++) {

						        		    #pragma vector always
						        		    #pragma ivdep
                                            for (z = myzbeg - (t - tv); z < myzbeg - (t - tv) + myzb ; z++) {
						        			    Compute_scalar(B, t, x, y, z);
                                            }
						        		}
						        	}
						        }

                                t = tv;

						        xmin += 1;
						        xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel] + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);

                                for (x = xmin; x < xmin + STRIDE + 1; x++) {

						        	for (y = myybeg - VECLEN + 1; y < myybeg - VECLEN + 1 + myyb; y++) {

						        	    for (z = myzbeg - VECLEN + 1; z <= myzbeg - VECLEN + 1 + myzb - VECLEN; z += VECLEN) {

                                            vload(v_center_0, B[t%2    ][x + STRIDE *3][y+3][z+3]);
                                            vload(v_center_1, B[(t+1)%2][x + STRIDE *2][y+2][z+2]);
                                            vload(v_center_2, B[t%2    ][x + STRIDE *1][y+1][z+1]);
                                            vload(v_center_3, B[(t+1)%2][x            ][y  ][z  ]);

                                            transpose(v_center_0, v_center_1, v_center_2, v_center_3, in, out);

                                            _mm256_storeu_pd(&Btmp[x - xmin][y - (myybeg - VECLEN + 1)][z - (myzbeg - VECLEN + 1) + 0][0], v_center_0);
                                            _mm256_storeu_pd(&Btmp[x - xmin][y - (myybeg - VECLEN + 1)][z - (myzbeg - VECLEN + 1) + 1][0], v_center_1);
                                            _mm256_storeu_pd(&Btmp[x - xmin][y - (myybeg - VECLEN + 1)][z - (myzbeg - VECLEN + 1) + 2][0], v_center_2);
                                            _mm256_storeu_pd(&Btmp[x - xmin][y - (myybeg - VECLEN + 1)][z - (myzbeg - VECLEN + 1) + 3][0], v_center_3);

						        	    }
                                    }
						        }

                                for (x = xmin + XSLOPE; x < xmax - STRIDE * VECLEN; x++) {

                                    for (y = myybeg - VECLEN + 1; y < myybeg - VECLEN + 1 + myyb; y++) {
                                       
                                        z = myzbeg - VECLEN + 1;

                                        v_center_0 = load_v(x, y, z, 0, 0, -1);

                                        if(y < myybeg - VECLEN + 1 + myyb - 1)
                                            v_center_3 = load_v(x + 2, y, z, 0, 1, 0); 

                                        v_center_1 = load_x_c_blocking(x, y, z, 0, 0);                                        

                                        v_all_d_1_0 = Add_4_vectors(    load_v(x, y, z, -1, 0, -1),\
                                                                        load_v(x, y, z, 1, 0, -1),\
                                                                        load_v(x, y, z, 0, 1, -1),\
                                                                        load_v(x, y, z, 0, -1, -1));
                                        v_all_d_2_0 = Add_4_vectors(    load_v(x, y, z, -1, 1, -1),\
                                                                        load_v(x, y, z, -1, -1, -1),\
                                                                        load_v(x, y, z, 1, 1, -1),\
                                                                        load_v(x, y, z, 1, -1, -1));

                                        
                                        v_all_d_1_1 = Add_4_d_1(x, y, z, 0);
                                        v_all_d_2_1 = Add_4_d_2(x, y, z, 0);

                                        for (; z <= myzbeg - VECLEN + 1 + myzb - VECLEN; z += VECLEN) {
                                            
                                            vload(in, B[(t)%2][x+STRIDE*4][y+VECLEN][z+VECLEN]);
                                            if(y < myybeg - VECLEN + 1 + myyb-1) {
                                                store_x_pp(v_center_3, x, y, z, 1, 0);
                                            }

                                            v_center_2 = load_x_c_blocking(x, y, z, 0, 1);
                                            v_all_d_1_2 = Add_4_d_1(x, y, z, 1);
                                            v_all_d_2_2 = Add_4_d_2(x, y, z, 1);
                                            Compute_1vector(v_center_0, \
                                                            v_center_1, \
                                                            v_center_2, \
                                                            v_center_3, \
                                                            v_all_d_1_0, \
                                                            v_all_d_1_1, \
                                                            v_all_d_1_2, \
                                                            v_all_d_2_0, \
                                                            v_all_d_2_1, \
                                                            v_all_d_2_2);
                                            if(y < myybeg - VECLEN + 1 + myyb - 1) {
                                                Input_Output_1(out, v_center_0, in);
                                                store_x_pp(v_center_0, x, y, z, 1, 1);
                                            }
                                            else {
                                                store_v(v_center_0, x, y, z, 0, 0, 0);
                                            }
                                            //-------------------------------------------------------------------------------------------

                                            v_center_3 = load_x_c_blocking(x, y, z, 0, 2);
                                            v_all_d_1_0 = Add_4_d_1(x, y, z, 2);
                                            v_all_d_2_0 = Add_4_d_2(x, y, z, 2);
                                            Compute_1vector(v_center_1, \
                                                            v_center_2, \
                                                            v_center_3, \
                                                            v_center_0, \
                                                            v_all_d_1_1, \
                                                            v_all_d_1_2, \
                                                            v_all_d_1_0, \
                                                            v_all_d_2_1, \
                                                            v_all_d_2_2, \
                                                            v_all_d_2_0);
                                            if(y < myybeg - VECLEN + 1 + myyb - 1) {
                                                Input_Output_2(out, v_center_1, in);
                                                store_x_pp(v_center_1, x, y, z, 1, 2);
                                            }
                                            else{
                                                store_v(v_center_1, x, y, z, 0, 0, 1);
                                            }
                                            
                                            // //-------------------------------------------------------------------------------------------
                        
                                            v_center_0 = load_x_c_blocking(x, y, z, 0, 3);
                                            v_all_d_1_1 = Add_4_d_1(x, y, z, 3);
                                            v_all_d_2_1 = Add_4_d_2(x, y, z, 3);
                                            Compute_1vector(v_center_2, \
                                                            v_center_3, \
                                                            v_center_0, \
                                                            v_center_1, \
                                                            v_all_d_1_2, \
                                                            v_all_d_1_0, \
                                                            v_all_d_1_1, \
                                                            v_all_d_2_2, \
                                                            v_all_d_2_0, \
                                                            v_all_d_2_1);	
                                            if(y < myybeg - VECLEN + 1 + myyb - 1) {
                                                Input_Output_3(out, v_center_2, in);
                                                store_x_pp(v_center_2, x, y, z, 1, 3); 
                                            }
                                            else{
                                                store_v(v_center_2, x, y, z, 0, 0, 2);
                                            }
                                            
                                            // //-------------------------------------------------------------------------------------------
                                            if ( z > myzbeg - VECLEN + 1 + myzb - VECLEN - VECLEN ){
                                                v_center_1 =  load_v(x, y, z, 0, 0, VECLEN);
                                                v_all_d_1_2 = Add_4_vectors(    load_v(x, y, z, -1, 0, VECLEN),\
                                                                                load_v(x, y, z, 1, 0, VECLEN),\
                                                                                load_v(x, y, z, 0, 1, VECLEN),\
                                                                                load_v(x, y, z, 0, -1, VECLEN));
                                                v_all_d_2_2 = Add_4_vectors(    load_v(x, y, z, -1, 1, VECLEN),\
                                                                                load_v(x, y, z, -1, -1, VECLEN),\
                                                                                load_v(x, y, z, 1, 1, VECLEN),\
                                                                                load_v(x, y, z, 1, -1, VECLEN));
                                            } else{
                                                v_center_1 = load_x_c_blocking(x, y, z, 0, 4);
                                                v_all_d_1_2 = Add_4_d_1(x, y, z, 4);
                                                v_all_d_2_2 = Add_4_d_2(x, y, z, 4);
                                            }


                                            Compute_1vector(v_center_3, \
                                                            v_center_0, \
                                                            v_center_1, \
                                                            v_center_2, \
                                                            v_all_d_1_0, \
                                                            v_all_d_1_1, \
                                                            v_all_d_1_2, \
                                                            v_all_d_2_0, \
                                                            v_all_d_2_1, \
                                                            v_all_d_2_2);
                                            if(y < myybeg - VECLEN + 1 + myyb - 1) {
                                                if (z >  myzbeg - VECLEN + 1 + myzb - VECLEN - VECLEN) {
									            	store_v(v_center_3, x, y, z, 0, 0, 3);
									            }
                                                Input_Output_4(out, v_center_3, in);	
                                                vstore(B[(t)%2][x][y][z], out);
                                            }
                                            else{
                                                store_v(v_center_3, x, y, z, 0, 0, 3);
                                            }

                                            v_all_d_1_0 = v_all_d_1_1;
                                            v_all_d_2_0 = v_all_d_2_1;

                                            v_all_d_1_1 = v_all_d_1_2;
                                            v_all_d_2_1 = v_all_d_2_2;
                                            //-------------------------------------------------------------------------------------------                                           

                                        }
                                        if(z >= myzbeg - VECLEN + 1 + myzb && y < myybeg - VECLEN + 1 + myyb - 1){


								        	B[(t ) % 2   ][x-1 + STRIDE * 3][y+3][z+2] = BV0[y - (myybeg - VECLEN + 1)][-1 + z - (myzbeg - VECLEN + 1)][0];
								        	B[(t + 1) % 2][x-1 + STRIDE * 2][y+2][z+1] = BV0[y - (myybeg - VECLEN + 1)][-1 + z - (myzbeg - VECLEN + 1)][1];
								        	B[(t) % 2    ][x-1 + STRIDE * 1][y+1][z  ] = BV0[y - (myybeg - VECLEN + 1)][-1 + z - (myzbeg - VECLEN + 1)][2];
								        	B[(t + 1) % 2][x-1             ][y  ][z-1] = BV0[y - (myybeg - VECLEN + 1)][-1 + z - (myzbeg - VECLEN + 1)][3];
								        }
                                        for ( ; z < myzbeg - VECLEN + 1 + myzb; z++) {
                                            v_all_d_1_2 = Add_4_vectors(    load_v(x, y, z, -1, 0, 1),\
                                                                            load_v(x, y, z, 1, 0, 1),\
                                                                            load_v(x, y, z, 0, 1, 1),\
                                                                            load_v(x, y, z, 0, -1, 1));
                                            v_all_d_2_2 = Add_4_vectors(    load_v(x, y, z, -1, 1, 1),\
                                                                            load_v(x, y, z, -1, -1, 1),\
                                                                            load_v(x, y, z, 1, 1, 1),\
                                                                            load_v(x, y, z, 1, -1, 1));
                                            v_center_2 = load_v(x, y, z, 0, 0, 1);
                                            Compute_1vector(v_center_0, \
                                                            v_center_1, \
                                                            v_center_2, \
                                                            v_center_3, \
                                                            v_all_d_1_0, \
                                                            v_all_d_1_1, \
                                                            v_all_d_1_2, \
                                                            v_all_d_2_0, \
                                                            v_all_d_2_1, \
                                                            v_all_d_2_2);
                                            store_v(v_center_0, x, y, z, 0, 0, 0);
                                            v_center_0 = v_center_1;
                                            v_center_1 = v_center_2;
                                            v_all_d_1_0 = v_all_d_1_1;
                                            v_all_d_2_0 = v_all_d_2_1;
                                            v_all_d_1_1 = v_all_d_1_2;
                                            v_all_d_2_1 = v_all_d_2_2;
                                        }
                                    }
                                    y = myybeg - VECLEN + 1;
                                    for (z = myzbeg - VECLEN + 1; z <= myzbeg - VECLEN + 1 + myzb - VECLEN; z += VECLEN) {

                                        vload(v_center_0, B[t%2    ][x + XSLOPE + 1 + STRIDE *3][y+3][z+3]);
                                        vload(v_center_1, B[(t+1)%2][x + XSLOPE + 1 + STRIDE *2][y+2][z+2]);
                                        vload(v_center_2, B[t%2    ][x + XSLOPE + 1 + STRIDE *1][y+1][z+1]);
                                        vload(v_center_3, B[(t+1)%2][x + XSLOPE + 1            ][y  ][z  ]);

                                        transpose(v_center_0, v_center_1, v_center_2, v_center_3, in, out);

                                        vstore( BV3[0][z - (myzbeg - VECLEN + 1) + 3][0], v_center_3);
                                        vstore( BV3[0][z - (myzbeg - VECLEN + 1) + 2][0], v_center_2);
                                        vstore( BV3[0][z - (myzbeg - VECLEN + 1) + 1][0], v_center_1);
                                        vstore( BV3[0][z - (myzbeg - VECLEN + 1) + 0][0], v_center_0);

						        	}
                                    y = myybeg - VECLEN + 1 + myyb - 1;
                                    for (z = myzbeg - VECLEN + 1; z <= myzbeg - VECLEN + 1 + myzb - VECLEN; z += VECLEN) {

                                        vload(v_center_3, BV0[myyb - 1][z - (myzbeg - VECLEN + 1) + 3][0]);
		                        		vload(v_center_2, BV0[myyb - 1][z - (myzbeg - VECLEN + 1) + 2][0]);
		                        		vload(v_center_1, BV0[myyb - 1][z - (myzbeg - VECLEN + 1) + 1][0]);
		                        		vload(v_center_0, BV0[myyb - 1][z - (myzbeg - VECLEN + 1) + 0][0]);

                                        transpose(v_center_0, v_center_1, v_center_2, v_center_3, in, out);

                                        vstore( B[t%2    ][x - XSLOPE + STRIDE *3][y+3][z+3], v_center_0);
		                        		vstore( B[(t+1)%2][x - XSLOPE + STRIDE *2][y+2][z+2], v_center_1);
		                        		vstore( B[t%2    ][x - XSLOPE + STRIDE *1][y+1][z+1], v_center_2);
		                        		vstore( B[(t+1)%2][x - XSLOPE            ][y  ][z  ], v_center_3);
						        	}
                                    Btmp[0] = BV0;                              
                                    BV0 = BV1;
                                    BV1 = BV2;
                                    BV2 = BV3;
                                    BV3 = Btmp[0];
                                }
                                Btmp [0] = BV0;
                                Btmp [1] = BV1;
                                Btmp [2] = BV2;

                                for (; x < xmax - STRIDE * VECLEN + 3; x++) {

		                        	for ( y = myybeg - VECLEN + 1; y < myybeg - VECLEN + 1 + myyb; y++) {

                                        for (z = myzbeg - VECLEN + 1; z <= myzbeg - VECLEN + 1 + myzb - VECLEN; z += VECLEN) {
		                        		    vload(v_center_3, Btmp[x - (xmax - STRIDE * VECLEN)][y - (myybeg - VECLEN + 1)][z - (myzbeg - VECLEN + 1) + 3][0]);
		                        		    vload(v_center_2, Btmp[x - (xmax - STRIDE * VECLEN)][y - (myybeg - VECLEN + 1)][z - (myzbeg - VECLEN + 1) + 2][0]);
		                        		    vload(v_center_1, Btmp[x - (xmax - STRIDE * VECLEN)][y - (myybeg - VECLEN + 1)][z - (myzbeg - VECLEN + 1) + 1][0]);
		                        		    vload(v_center_0, Btmp[x - (xmax - STRIDE * VECLEN)][y - (myybeg - VECLEN + 1)][z - (myzbeg - VECLEN + 1) + 0][0]);

		                        		    transpose(v_center_0, v_center_1, v_center_2, v_center_3, in, out);

		                        		    vstore( B[t%2    ][x - XSLOPE + STRIDE *3][y+3][z+3], v_center_0);
		                        		    vstore( B[(t+1)%2][x - XSLOPE + STRIDE *2][y+2][z+2], v_center_1);
		                        		    vstore( B[t%2    ][x - XSLOPE + STRIDE *1][y+1][z+1], v_center_2);
		                        		    vstore( B[(t+1)%2][x - XSLOPE            ][y  ][z  ], v_center_3);
                                        }
		                        	}
		                        }

                                xmin = xmax - STRIDE * VECLEN;
						        for (t = tv; t < tv + VECLEN; t++) {

							        xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel] + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);

							        for (x = xmin + STRIDE * (VECLEN - 1 - (t - tv)); x < xmax; x++) {

						        	    for (y = myybeg - (t - tv); y < myybeg - (t - tv) + myyb; y++) {
						        	        #pragma ivdep
						        	        #pragma vector always
						        	    	for (z = myzbeg - (t - tv); z < myzbeg - (t - tv) + myzb ; z++) {
                                                Compute_scalar(B, t, x, y, z);
                                            }
						        	    }
                                    }
						        }

                                for (t = tv; t < tv + VECLEN; t++){

						    	    xmin = (mylevel == 1 && xx == 0) ?	             XSTART : (xright[mylevel] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
						            xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel]      + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);

                                    for (x = xmin; x < xmax; x++) {

                                        for ( y = myybeg + myyb - (t - tv); y < min(NY + YSTART, ybeg + yb - (t - tv)); y++) {								
								    		#pragma vector always
						        		    #pragma ivdep

                                            for (z = myzbeg - (t - tv); z < myzbeg - (t - tv) + myzb; z++)
								    		{
								    			Compute_scalar(B, t, x, y, z);
								    		}
								    	}
								    }
								    for (x = xmin; x < xmax; x++) {

                                        for ( y = max(YSTART, ybeg - (t - tv)); y < min(NY + YSTART, ybeg + yb - (t - tv)); y++){
								    		#pragma vector always
						        		    #pragma ivdep

                                            for (z = myzbeg + myzb - (t - tv); z < min(NZ + ZSTART, zbeg + zb - (t - tv)); z++)
								    		{
								    			Compute_scalar(B, t, x, y, z);
								    		}
								    	}
								    }
						        }
                            }
                        }
                        //extra
                        //
                        for (t = tv; t < min(tt + 2 * tb, T); t++) {
                        
					    	xmin = (mylevel == 1 && xx == 0) ?	             XSTART : (xright[mylevel] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
					    	xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel]      + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);
    
					    	for (x = xmin; x < xmax; x++) {

					    		for (y = max(YSTART, ybeg - (t - tv)); y < min(NY + YSTART, ybeg - (t - tv) + yb); y++) {
                                    #pragma ivdep
					    		    #pragma vector always
                                    for (z = max(ZSTART, zbeg - (t - tv)); z < min(NZ + ZSTART, zbeg - (t - tv) + zb); z++){
					    			    Compute_scalar(B, t, x, y, z);
                                    }
					    		}
					    	}
					    }
					}
				}				
			}
		}
	}
    free_extra_array(AV);
 #ifdef scalar_ratio
	printf("%f\n", (double) cnt /(double)((double)NX * (double) NY *(double) NZ * (double) T));
#endif   
}
