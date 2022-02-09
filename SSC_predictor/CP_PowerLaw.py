# Crystal Plasticity Code by Shahriyar Keshavarz @ NIST

import math, sys
from pylab import *
import time
import matplotlib.pyplot as plt

########## Loading gauss points for 8 integration points per element ############
def pgauss():
    g = 1.0 / math.sqrt(3.0)
    ls = [-1, 1, 1, -1, -1, 1, 1, -1]
    lt = [-1, -1, 1, 1, -1, -1, 1, 1]
    lz = [-1, -1, -1, -1, 1, 1, 1, 1]
    sg = [];
    tg = [];
    zg = [];
    wg = []
    for i in range(8):
        s = g * ls[i];
        sg.append(s)
        t = g * lt[i];
        tg.append(t)
        z = g * lz[i];
        zg.append(z)
        w = 1.0;
        wg.append(w)
    return sg, tg, zg, wg


##################### End of loading gauss points ################################

##### Get global shape functions and their derivatives at a given gauss point#####
############################ N_i(X,t) and gradN_i(X,t) ###########################
def SHP3D(ss, tt, zz, XL=[[]]):
    si = [-1, 1, 1, -1, -1, 1, 1, -1]
    ti = [-1, -1, 1, 1, -1, -1, 1, 1]
    zi = [-1, -1, -1, -1, 1, 1, 1, 1]
    SHPL = [[0. for ii in range(8)] for jj in range(4)]
    # ----------- Local shape functions with derivatives with repect to si,ti and zi
    for i in range(8):
        N1 = 1.0 + ss * si[i]
        N2 = 1.0 + tt * ti[i]
        N3 = 1.0 + zz * zi[i]
        SHPL[3][i] = N1 * N2 * N3 / 8.0
        SHPL[0][i] = si[i] * N2 * N3 / 8.0
        SHPL[1][i] = ti[i] * N1 * N3 / 8.0
        SHPL[2][i] = zi[i] * N1 * N2 / 8.0
    # --------- Get global values in refrence configuration

    XS = [[0. for ii in range(3)] for jj in range(3)]
    SX = [[0. for ii in range(3)] for jj in range(3)]

    for i in range(3):
        for j in range(3):
            for k in range(8):
                XS[i][j] += XL[k][j] * SHPL[i][k]

    SXD = matinv3(XS)
    SX = SXD[0];
    xsJ = SXD[1]

    SHPG = [[0. for ii in range(8)] for jj in range(3)]
    for i in range(8):
        T0 = SX[0][0] * SHPL[0][i] + SX[0][1] * SHPL[1][i] + SX[0][2] * SHPL[2][i]
        T1 = SX[1][0] * SHPL[0][i] + SX[1][1] * SHPL[1][i] + SX[1][2] * SHPL[2][i]
        T2 = SX[2][0] * SHPL[0][i] + SX[2][1] * SHPL[1][i] + SX[2][2] * SHPL[2][i]
        SHPG[0][i] = T0
        SHPG[1][i] = T1
        SHPG[2][i] = T2

    return SHPG, xsJ


############### End of calculation of shape functions and drivatives ##################

########### Calculate deformation gradient (dx/dX=sum(x*gradN_i(X,t))) ################
def calc_F(cord, SHP):
    dxt_dx0 = 0.0;
    dxt_dy0 = 0.0;
    dxt_dz0 = 0.0
    dyt_dx0 = 0.0;
    dyt_dy0 = 0.0;
    dyt_dz0 = 0.0
    dzt_dx0 = 0.0;
    dzt_dy0 = 0.0;
    dzt_dz0 = 0.0

    for I in range(8):
        dxt_dx0 += SHP[0][I] * cord[I][0]
        dxt_dy0 += SHP[1][I] * cord[I][0]
        dxt_dz0 += SHP[2][I] * cord[I][0]

        dyt_dx0 += SHP[0][I] * cord[I][1]
        dyt_dy0 += SHP[1][I] * cord[I][1]
        dyt_dz0 += SHP[2][I] * cord[I][1]

        dzt_dx0 += SHP[0][I] * cord[I][2]
        dzt_dy0 += SHP[1][I] * cord[I][2]
        dzt_dz0 += SHP[2][I] * cord[I][2]

    FG = [[0. for ii in range(3)] for jj in range(3)]
    FG[0][0] = dxt_dx0;
    FG[0][1] = dxt_dy0;
    FG[0][2] = dxt_dz0
    FG[1][0] = dyt_dx0;
    FG[1][1] = dyt_dy0;
    FG[1][2] = dyt_dz0
    FG[2][0] = dzt_dx0;
    FG[2][1] = dzt_dy0;
    FG[2][2] = dzt_dz0

    return FG


#################### End of calculation of deformation gradient ###################

############ Calculate A(n,n)*B(n,n)############
# def matmul(len,A=[[]],B=[[]]):

def matmul(A, B):
    A = list(A)
    B = list(B)
    C = [[0. for ii in range(len(A))] for jj in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A)):
            for k in range(len(A)):
                C[i][j] += A[i][k] * B[k][j]
    return C


###########################################################################
##################### Schmid Tensor for FCC materials #####################
###########################################################################

def calc_schmid(crystal_type, n_slip, qrot):
    schmid = [[[0. for i in range(3)] for j in range(3)] for k in range(n_slip)]

    r1 = 1.0
    r2 = 1.0 / math.sqrt(2.0)
    r3 = 1.0 / math.sqrt(3.0)
    r6 = 1.0 / math.sqrt(6.0)

    if crystal_type is 'fcc':

        cns = [[0. for ii in range(3)] for jj in range(n_slip)]
        cms = [[0. for ii in range(3)] for jj in range(n_slip)]
        vec1 = [[0. for ii in range(3)] for jj in range(1)]
        vec2 = [[0. for ii in range(1)] for jj in range(3)]

        # ---------------fcc slip system 1----------
        ind_slip = 0

        cns[ind_slip][0] = r3
        cns[ind_slip][1] = r3
        cns[ind_slip][2] = r3

        cms[ind_slip][0] = 0.0
        cms[ind_slip][1] = r2
        cms[ind_slip][2] = -r2

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

        # ---------------fcc slip system 2----------
        ind_slip = 1

        cns[ind_slip][0] = r3
        cns[ind_slip][1] = r3
        cns[ind_slip][2] = r3

        cms[ind_slip][0] = -r2
        cms[ind_slip][1] = 0.0
        cms[ind_slip][2] = r2

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

        # ---------------fcc slip system 3----------
        ind_slip = 2

        cns[ind_slip][0] = r3
        cns[ind_slip][1] = r3
        cns[ind_slip][2] = r3

        cms[ind_slip][0] = r2
        cms[ind_slip][1] = -r2
        cms[ind_slip][2] = 0.0

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

        # ---------------fcc slip system 4----------
        ind_slip = 3

        cns[ind_slip][0] = r3
        cns[ind_slip][1] = -r3
        cns[ind_slip][2] = -r3

        cms[ind_slip][0] = 0.0
        cms[ind_slip][1] = -r2
        cms[ind_slip][2] = r2

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

        # ---------------fcc slip system 5----------
        ind_slip = 4

        cns[ind_slip][0] = r3
        cns[ind_slip][1] = -r3
        cns[ind_slip][2] = -r3

        cms[ind_slip][0] = -r2
        cms[ind_slip][1] = 0.0
        cms[ind_slip][2] = -r2

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

        # ---------------fcc slip system 6----------
        ind_slip = 5

        cns[ind_slip][0] = r3
        cns[ind_slip][1] = -r3
        cns[ind_slip][2] = -r3

        cms[ind_slip][0] = r2
        cms[ind_slip][1] = r2
        cms[ind_slip][2] = 0.0

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

        # ---------------fcc slip system 7----------
        ind_slip = 6

        cns[ind_slip][0] = -r3
        cns[ind_slip][1] = r3
        cns[ind_slip][2] = -r3

        cms[ind_slip][0] = 0.0
        cms[ind_slip][1] = r2
        cms[ind_slip][2] = r2

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

        # ---------------fcc slip system 8----------
        ind_slip = 7

        cns[ind_slip][0] = -r3
        cns[ind_slip][1] = r3
        cns[ind_slip][2] = -r3

        cms[ind_slip][0] = r2
        cms[ind_slip][1] = 0.0
        cms[ind_slip][2] = -r2

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

        # ---------------fcc slip system 9----------
        ind_slip = 8

        cns[ind_slip][0] = -r3
        cns[ind_slip][1] = r3
        cns[ind_slip][2] = -r3

        cms[ind_slip][0] = -r2
        cms[ind_slip][1] = -r2
        cms[ind_slip][2] = 0.0

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

        # ---------------fcc slip system 10----------
        ind_slip = 9

        cns[ind_slip][0] = -r3
        cns[ind_slip][1] = -r3
        cns[ind_slip][2] = r3

        cms[ind_slip][0] = 0.0
        cms[ind_slip][1] = -r2
        cms[ind_slip][2] = -r2

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

        # ---------------fcc slip system 11----------
        ind_slip = 10

        cns[ind_slip][0] = -r3
        cns[ind_slip][1] = -r3
        cns[ind_slip][2] = r3

        cms[ind_slip][0] = r2
        cms[ind_slip][1] = 0.0
        cms[ind_slip][2] = r2

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

        # ---------------fcc slip system 12----------
        ind_slip = 11

        cns[ind_slip][0] = -r3
        cns[ind_slip][1] = -r3
        cns[ind_slip][2] = r3

        cms[ind_slip][0] = -r2
        cms[ind_slip][1] = r2
        cms[ind_slip][2] = 0.0

        for ij in range(3):
            vec2[ij][0] = cns[ind_slip][ij]

        for ik in range(3):
            vec1[0][ik] = cms[ind_slip][ik]

        tmp_v1 = [[0. for ii in range(3)] for jj in range(1)]
        tmp_v2 = [[0. for ii in range(1)] for jj in range(3)]
        tmp1 = [[0. for ii in range(3)] for jj in range(3)]

        for i in range(3):
            for j in range(3):
                tmp_v1[0][i] += qrot[i][j] * vec1[0][j]
                tmp_v2[i][0] += qrot[i][j] * vec2[j][0]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = tmp_v1[0][i] * tmp_v2[j][0]

        for i in range(3):
            for j in range(3):
                schmid[ind_slip][i][j] = tmp1[i][j]

    return schmid


############## End of Schmid Tensor for FCC materials #####################


############## Rotation Matrix based on Euler angles #####################

def euler_rot(angle_el, qrot):
    const_pi = math.acos(-1.0)

    phi = angle_el[0] * const_pi / 180.0
    theta = angle_el[1] * const_pi / 180.0
    omega = angle_el[2] * const_pi / 180.0

    sp = math.sin(phi)
    cp = math.cos(phi)
    st = math.sin(theta)
    ct = math.cos(theta)
    so = math.sin(omega)
    co = math.cos(omega)

    qrot[0][0] = co * cp - so * sp * ct
    qrot[1][0] = co * sp + so * ct * cp
    qrot[2][0] = so * st
    qrot[0][1] = -so * cp - sp * co * ct
    qrot[1][1] = -so * sp + ct * co * cp
    qrot[2][1] = co * st
    qrot[0][2] = sp * st
    qrot[1][2] = -st * cp
    qrot[2][2] = ct

    return qrot


###################### End of Rotation Matrix #####################

############## Elastic Material 2nd and 4th order tensor ###################

def cmat(c11, c12, c44, qrot):
    cij = [[0. for i in range(6)] for j in range(6)]

    # ---------------- cij for cubic materials
    cij[0][0] = cij[1][1] = cij[2][2] = c11
    cij[0][1] = cij[0][2] = cij[1][0] = cij[1][2] = cij[2][0] = cij[2][1] = c12
    cij[3][3] = cij[4][4] = cij[5][5] = c44

    cijkl = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]

    # ----------------- Get cijkl
    for i in range(3):
        for j in range(3):
            cijkl[i][i][j][j] = cij[i][j]

    for i in range(3):
        cijkl[i][i][0][1] = cij[i][3]
        cijkl[i][i][1][0] = cij[i][3]
        cijkl[i][i][1][2] = cij[i][4]
        cijkl[i][i][2][1] = cij[i][4]
        cijkl[i][i][0][2] = cij[i][5]
        cijkl[i][i][2][0] = cij[i][5]

        cijkl[0][1][i][i] = cijkl[1][0][i][i] = cij[3][i]
        cijkl[1][2][i][i] = cijkl[2][1][i][i] = cij[4][i]
        cijkl[0][2][i][i] = cijkl[2][0][i][i] = cij[5][i]

    cijkl[0][1][0][1] = cijkl[0][1][1][0] = cijkl[1][0][0][1] = cijkl[1][0][1][0] = cij[3][3]
    cijkl[0][1][1][2] = cijkl[0][1][2][1] = cijkl[1][0][1][2] = cijkl[1][0][2][1] = cij[3][4]
    cijkl[0][1][0][2] = cijkl[0][1][2][0] = cijkl[1][0][0][2] = cijkl[1][0][2][0] = cij[3][5]
    cijkl[1][2][0][1] = cijkl[1][2][1][0] = cijkl[2][1][0][1] = cijkl[2][1][1][0] = cij[4][3]
    cijkl[1][2][1][2] = cijkl[1][2][2][1] = cijkl[2][1][1][2] = cijkl[2][1][2][1] = cij[4][4]
    cijkl[1][2][0][2] = cijkl[1][2][2][0] = cijkl[2][1][0][2] = cijkl[2][1][2][0] = cij[4][5]
    cijkl[0][2][0][1] = cijkl[0][2][1][0] = cijkl[2][0][0][1] = cijkl[2][0][1][0] = cij[5][3]
    cijkl[0][2][1][2] = cijkl[0][2][2][1] = cijkl[2][0][1][2] = cijkl[2][0][2][1] = cij[5][4]
    cijkl[0][2][0][2] = cijkl[0][2][2][0] = cijkl[2][0][0][2] = cijkl[2][0][2][0] = cij[5][5]

    # ----------------- C_MAT for Crytal with respect to orientation
    C_mat = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):

                    for m in range(3):
                        for n in range(3):
                            for p in range(3):
                                for q in range(3):
                                    C_mat[i][j][k][l] += qrot[i][m] * qrot[j][n] * qrot[k][p] * qrot[l][q] * \
                                                         cijkl[m][n][p][q]

    return C_mat


########################## End of calculation C_mat ##########################

########################## Calculate matrix [A] ##############################
def calc_A(F, Fp):
    F_tau = [[0. for ii in range(3)] for jj in range(3)]
    FT_tau = [[0. for ii in range(3)] for jj in range(3)]
    FpI_t = [[0. for ii in range(3)] for jj in range(3)]
    FpIT_t = [[0. for ii in range(3)] for jj in range(3)]
    TMP1 = [[0. for ii in range(3)] for jj in range(3)]
    TMP2 = [[0. for ii in range(3)] for jj in range(3)]
    A_mat = [[0. for ii in range(3)] for jj in range(3)]

    FpID = matinv3(Fp)
    FpI_t = FpID[0];
    FpIT_t = zip(*FpI_t)

    for i in range(3):
        for j in range(3):
            F_tau[i][j] = F[i][j]
            FT_tau[i][j] = F[j][i]

    TMP1 = matmul(F_tau, FpI_t)
    TMP2 = matmul(FT_tau, TMP1)
    A_mat = matmul(FpIT_t, TMP2)

    return A_mat


########################## End of A_mat ##################################

################ Calculate matrix [B_alpha] ##############################
def calc_B(n_slip, A_mat, schmid):
    B_alpha = [[[0. for i in range(3)] for j in range(3)] for k in range(n_slip)]
    TMP1 = [[0. for ii in range(3)] for jj in range(3)]
    TMP1T = [[0. for ii in range(3)] for jj in range(3)]
    TMP20 = [[0. for ii in range(3)] for jj in range(3)]
    TMP21 = [[0. for ii in range(3)] for jj in range(3)]
    TMP2 = [[0. for ii in range(3)] for jj in range(3)]
    for k in range(n_slip):
        for j in range(3):
            for i in range(3):
                TMP1[i][j] = schmid[k][i][j]
                TMP1T[i][j] = schmid[k][j][i]

        TMP20 = matmul(A_mat, TMP1)
        TMP21 = matmul(TMP1T, A_mat)

        for it in range(3):
            for jt in range(3):
                TMP2[it][jt] = TMP20[it][jt] + TMP21[it][jt]

        for i in range(3):
            for j in range(3):
                B_alpha[k][i][j] = TMP2[i][j]

    return B_alpha


###################### End of B_alpha #####################################

############# Calculate the trial elastic stress S_trial ##################
def stress_trial(C_mat, A_mat):
    delta_kron = [[0. for ii in range(3)] for jj in range(3)]
    S_trial = [[0. for ii in range(3)] for jj in range(3)]

    delta_kron[0][0] = delta_kron[1][1] = delta_kron[2][2] = 1.0

    for i in range(3):
        for j in range(3):
            for m in range(3):
                for n in range(3):
                    S_trial[i][j] += C_mat[i][j][m][n] * 0.5 * (A_mat[m][n] - delta_kron[m][n])

    return S_trial


########################## End of S_trial ####################################

################ Calculate the matrix C_ALPHA for each slip system ###########
def calc_C_alpha(n_slip, C_mat, B_alpha):
    C_alpha = [[[0. for i in range(3)] for j in range(3)] for k in range(n_slip)]

    for k in range(n_slip):
        for i in range(3):
            for j in range(3):
                for m in range(3):
                    for n in range(3):
                        C_alpha[k][i][j] += C_mat[i][j][m][n] * 0.5 * B_alpha[k][m][n]

    return C_alpha


####################### End of C_alpha ####################################

################  Resolved Shear Stress ##########
def resolvedshear(n_slip, stress, schmidfactor):
    tau_alpha = [0.0 for k in range(n_slip)]
    for k in range(n_slip):
        for i in range(3):
            for j in range(3):
                tau_alpha[k] += stress[i][j] * schmidfactor[k][i][j]

    return tau_alpha


#################### End of Resolved Shear Stress ##############

#################  Reduce 4th order 3*3*3*3 tensor to 2nd 6*6 matrix ########

def reduce_mat(as_mat):
    as_redu = [[0.0 for i in range(6)] for j in range(6)]

    for i in range(3):
        for j in range(3):
            as_redu[i][j] = as_mat[i][i][j][j]

    for i in range(3):
        as_redu[i][3] = as_mat[i][i][0][1] + as_mat[i][i][1][0]
        as_redu[i][4] = as_mat[i][i][1][2] + as_mat[i][i][2][1]
        as_redu[i][5] = as_mat[i][i][0][2] + as_mat[i][i][2][0]

    for j in range(3):
        as_redu[3][j] = as_mat[0][1][j][j]
        as_redu[4][j] = as_mat[1][2][j][j]
        as_redu[5][j] = as_mat[0][2][j][j]

    as_redu[3][3] = as_mat[0][1][0][1] + as_mat[0][1][1][0]
    as_redu[3][4] = as_mat[0][1][1][2] + as_mat[0][1][2][1]
    as_redu[3][5] = as_mat[0][1][0][2] + as_mat[0][1][2][0]

    as_redu[4][3] = as_mat[1][2][0][1] + as_mat[1][2][1][0]
    as_redu[4][4] = as_mat[1][2][1][2] + as_mat[1][2][2][1]
    as_redu[4][5] = as_mat[1][2][0][2] + as_mat[1][2][2][0]

    as_redu[5][3] = as_mat[0][2][0][1] + as_mat[0][2][1][0]
    as_redu[5][4] = as_mat[0][2][1][2] + as_mat[0][2][2][1]
    as_redu[5][5] = as_mat[0][2][0][2] + as_mat[0][2][2][0]

    return as_redu


#################  end of Reduce 4th order 3*3*3*3 tensor to 2nd 6*6 matrix ########

#################  Reduce 4th order 3*3*3*3 W_mat tensor to 2nd 6*6 matrix ########

def reduce_wmat(as_mat):
    as_redu = [[0.0 for i in range(6)] for j in range(6)]

    for i in range(3):
        for j in range(3):
            as_redu[i][j] = as_mat[i][i][j][j]

    for i in range(3):
        as_redu[i][3] = (as_mat[i][i][0][1] + as_mat[i][i][1][0]) / 2.0
        as_redu[i][4] = (as_mat[i][i][1][2] + as_mat[i][i][2][1]) / 2.0
        as_redu[i][5] = (as_mat[i][i][0][2] + as_mat[i][i][2][0]) / 2.0

    for j in range(3):
        as_redu[3][j] = as_mat[0][1][j][j]
        as_redu[4][j] = as_mat[1][2][j][j]
        as_redu[5][j] = as_mat[0][2][j][j]

    as_redu[3][3] = (as_mat[0][1][0][1] + as_mat[0][1][1][0]) / 2.0
    as_redu[3][4] = (as_mat[0][1][1][2] + as_mat[0][1][2][1]) / 2.0
    as_redu[3][5] = (as_mat[0][1][0][2] + as_mat[0][1][2][0]) / 2.0

    as_redu[4][3] = (as_mat[1][2][0][1] + as_mat[1][2][1][0]) / 2.0
    as_redu[4][4] = (as_mat[1][2][1][2] + as_mat[1][2][2][1]) / 2.0
    as_redu[4][5] = (as_mat[1][2][0][2] + as_mat[1][2][2][0]) / 2.0

    as_redu[5][3] = (as_mat[0][2][0][1] + as_mat[0][2][1][0]) / 2.0
    as_redu[5][4] = (as_mat[0][2][1][2] + as_mat[0][2][2][1]) / 2.0
    as_redu[5][5] = (as_mat[0][2][0][2] + as_mat[0][2][2][0]) / 2.0

    return as_redu


#################  end of Reduce 4th order 3*3*3*3 tensor to 2nd 6*6 matrix ########

#################  Reduce 2nd order 3*3 tensor to 1st 6 vector ########

def reduce_vec(as_star):
    as_vec = [0.0 for i in range(6)]
    for i in range(3):
        as_vec[i] = as_star[i][i]

    as_vec[3] = 0.5 * (as_star[0][1] + as_star[1][0])
    as_vec[4] = 0.5 * (as_star[1][2] + as_star[2][1])
    as_vec[5] = 0.5 * (as_star[0][2] + as_star[2][0])

    return as_vec


#################  end of Reduce 2nd order 3*3 tensor to 1st 6 vector ########


######### Inflate 1st 6 vector to 2nd 3*3 tensor ################
def inflate_vec(as_vec):
    as_star = [[0.0 for i in range(3)] for j in range(3)]
    for i in range(3):
        as_star[i][i] = as_vec[i]

    as_star[0][1] = as_vec[3]
    as_star[1][2] = as_vec[4]
    as_star[0][2] = as_vec[5]

    as_star[1][0] = as_vec[3]
    as_star[2][1] = as_vec[4]
    as_star[2][0] = as_vec[5]

    return as_star


######### end of Inflate 1st 6 vector to 2nd 3*3 tensor ################

######### Inflate 6*6 matrix to 4rt order tenso ################
def inflate_ten(as_ten):
    as_mat = [[[[0.0 for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]
    for i in range(3):
        for j in range(3):
            as_mat[i][i][j][j] = as_ten[i][j]

    for i in range(3):
        as_mat[i][i][0][1] = as_mat[i][i][1][0] = 0.5 * as_ten[i][3]
        as_mat[i][i][1][2] = as_mat[i][i][2][1] = 0.5 * as_ten[i][4]
        as_mat[i][i][0][2] = as_mat[i][i][2][0] = 0.5 * as_ten[i][5]

    for j in range(3):
        as_mat[0][1][j][j] = as_mat[1][0][j][j] = as_ten[3][j]
        as_mat[1][2][j][j] = as_mat[2][1][j][j] = as_ten[4][j]
        as_mat[0][2][j][j] = as_mat[2][0][j][j] = as_ten[5][j]

    as_mat[0][1][0][1] = as_mat[0][1][1][0] = as_mat[1][0][0][1] = as_mat[1][0][1][0] = 0.5 * as_ten[3][3]
    as_mat[0][1][1][2] = as_mat[0][1][2][1] = as_mat[1][0][1][2] = as_mat[1][0][2][1] = 0.5 * as_ten[3][4]
    as_mat[0][1][0][2] = as_mat[0][1][2][0] = as_mat[1][0][0][2] = as_mat[1][0][2][0] = 0.5 * as_ten[3][5]

    as_mat[1][2][0][1] = as_mat[1][2][1][0] = as_mat[2][1][0][1] = as_mat[2][1][1][0] = 0.5 * as_ten[4][3]
    as_mat[1][2][1][2] = as_mat[1][2][2][1] = as_mat[2][1][1][2] = as_mat[2][1][2][1] = 0.5 * as_ten[4][4]
    as_mat[1][2][0][2] = as_mat[1][2][2][0] = as_mat[2][1][0][2] = as_mat[2][1][2][0] = 0.5 * as_ten[4][5]

    as_mat[0][2][0][1] = as_mat[0][2][1][0] = as_mat[2][0][0][1] = as_mat[2][0][1][0] = 0.5 * as_ten[5][3]
    as_mat[0][2][1][2] = as_mat[0][2][2][1] = as_mat[2][0][1][2] = as_mat[2][0][2][1] = 0.5 * as_ten[5][4]
    as_mat[0][2][0][2] = as_mat[0][2][2][0] = as_mat[2][0][0][2] = as_mat[2][0][2][0] = 0.5 * as_ten[5][5]

    return as_mat


############ end of Inflate of 6*6 matrix to 4rt order tensor ###########

################# Calculation of GN vector ###################

def calc_GNvec(n_slip, S_star, S_trial, dgam, C_alpha):
    GN_vec = [0.0 for i in range(6)]
    TMP1 = [[0.0 for i in range(3)] for j in range(3)]

    for i in range(3):
        for j in range(3):
            TMP1[i][j] = 0.0
            for k in range(n_slip):
                TMP1[i][j] += dgam[k] * C_alpha[k][i][j]

    for i in range(3):
        for j in range(3):
            TMP1[i][j] += S_star[i][j] - S_trial[i][j]

    GN_vec = reduce_vec(TMP1)

    return GN_vec


################# end of Calculation of GN vector ###################

################### element reaction or internal forces ##########
def element_re(ngauss, nnode, xyz, strs):
    InF = [0.0 for i in range(nnode * 3)]

    for ig in range(ngauss):
        gaussw = pgauss()
        SHP = SHP3D(gaussw[0][ig], gaussw[1][ig], gaussw[2][ig], xyz)
        xsJ = SHP[1]
        J1 = 0
        for ind in range(nnode):
            InF[J1] += (SHP[0][0][ind] * strs[ig][0] + SHP[0][1][ind] * strs[ig][3] + SHP[0][2][ind] * strs[ig][
                5]) * xsJ

            InF[J1 + 1] += (SHP[0][1][ind] * strs[ig][1] + SHP[0][0][ind] * strs[ig][3] + SHP[0][2][ind] * strs[ig][
                4]) * xsJ

            InF[J1 + 2] += (SHP[0][2][ind] * strs[ig][2] + SHP[0][1][ind] * strs[ig][4] + SHP[0][0][ind] * strs[ig][
                5]) * xsJ

            J1 += 3

    return InF


######################### end of element reaction forces #######################

############ making global internal forces  #############
def asembl_vec(gf_nod, InF, lnode):
    node_indx = [0 for i in range(24)]

    for i in range(8):
        node_indx[3 * i] = 3 * lnode[i];
        node_indx[3 * i + 1] = 3 * lnode[i] + 1;
        node_indx[3 * i + 2] = 3 * lnode[i] + 2

    for i in range(24):
        gf_nod[node_indx[i]] += InF[i]

    return gf_nod


############ end of making global internal forces  #############
############ dot product of a vector  #############
def dot_product(gu, neq):
    sf = 0.0
    for i in range(neq):
        sf += gu[i] * gu[i]

    return sf


############ end of dot product of a vector  #############


################### calculation of stiffness matrix ################
def STIF3D(ngauss, nnode, xyz, strs, dep):
    vol_gp = [0.0 for i in range(ngauss)]
    gaussw = pgauss()

    ES = [[0.0 for i in range(24)] for j in range(24)]
    ESM = [[0.0 for i in range(24)] for j in range(24)]
    ESGEOM = [[0.0 for i in range(24)] for j in range(24)]
    for ig in range(ngauss):

        SHP = SHP3D(gaussw[0][ig], gaussw[1][ig], gaussw[2][ig], xyz)
        xsj = SHP[1] * gaussw[3][ig]

        vol_gp[ig] = xsj

        B = [[0.0 for i in range(24)] for j in range(6)]
        for i in range(nnode):
            ii = i * 3
            B[0][ii + 0] = SHP[0][0][i];
            B[0][ii + 1] = 0.0;
            B[0][ii + 2] = 0.0
            B[1][ii + 0] = 0.0;
            B[1][ii + 1] = SHP[0][1][i];
            B[1][ii + 2] = 0.0
            B[2][ii + 0] = 0.0;
            B[2][ii + 1] = 0.0;
            B[2][ii + 2] = SHP[0][2][i]
            B[3][ii + 0] = SHP[0][1][i];
            B[3][ii + 1] = SHP[0][0][i];
            B[3][ii + 2] = 0.0
            B[4][ii + 0] = 0.0;
            B[4][ii + 1] = SHP[0][2][i];
            B[4][ii + 2] = SHP[0][1][i]
            B[5][ii + 0] = SHP[0][2][i];
            B[5][ii + 1] = 0.0;
            B[5][ii + 2] = SHP[0][0][i]

        DB = [[0.0 for i in range(24)] for j in range(6)]
        for i in range(6):
            for j in range(24):
                DB[i][j] = 0.0
                for k in range(6):
                    DB[i][j] += dep[ig][i][k] * B[k][j]

        for i in range(24):
            for j in range(24):
                for k in range(6):
                    ESM[i][j] += B[k][i] * DB[k][j] * xsj

        BNL = [[0.0 for i in range(24)] for j in range(9)]
        DBNL = [[0.0 for i in range(24)] for j in range(9)]
        stressgeo = [[0.0 for i in range(9)] for j in range(9)]

        for i in range(8):
            icol1 = i * 3 + 0
            icol2 = icol1 + 1
            icol3 = icol2 + 1

            BNL[0][icol1] = SHP[0][0][i]
            BNL[1][icol1] = SHP[0][1][i]
            BNL[2][icol1] = SHP[0][2][i]
            BNL[3][icol2] = SHP[0][0][i]
            BNL[4][icol2] = SHP[0][1][i]
            BNL[5][icol2] = SHP[0][2][i]
            BNL[6][icol3] = SHP[0][0][i]
            BNL[7][icol3] = SHP[0][1][i]
            BNL[8][icol3] = SHP[0][2][i]

        for i in range(0, 9, 3):
            stressgeo[i][i] = strs[ig][0]
            stressgeo[i][i + 1] = strs[ig][3]
            stressgeo[i][i + 2] = strs[ig][5]
            stressgeo[i + 1][i] = strs[ig][3]
            stressgeo[i + 1][i + 1] = strs[ig][1]
            stressgeo[i + 1][i + 2] = strs[ig][4]
            stressgeo[i + 2][i] = strs[ig][5]
            stressgeo[i + 2][i + 1] = strs[ig][4]
            stressgeo[i + 2][i + 2] = strs[ig][2]

        for i in range(9):
            for j in range(24):
                for k in range(9):
                    DBNL[i][j] += stressgeo[i][k] * BNL[k][j]

        for i in range(24):
            for j in range(24):
                for k in range(9):
                    ESGEOM[i][j] += BNL[k][i] * DBNL[k][j] * xsj

        for i in range(24):
            for j in range(24):
                ES[i][j] = ESM[i][j] + ESGEOM[i][j]

    return ES, vol_gp


################# end of element stiffness matrix calculation #########

################# inverse 3by3 matrix ######################
def matinv3(a):
    a_inv = [[0.0 for i in range(3)] for j in range(3)]

    det = a[0][0] * a[1][1] * a[2][2] - a[0][0] * a[1][2] * a[2][1] - a[1][0] * a[0][1] \
          * a[2][2] + a[1][0] * a[0][2] * a[2][1] + a[2][0] * a[0][1] * a[1][2] - a[2][0] \
          * a[0][2] * a[1][1]

    a_inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det
    a_inv[0][1] = -(a[0][1] * a[2][2] - a[0][2] * a[2][1]) / det
    a_inv[0][2] = -(-a[0][1] * a[1][2] + a[0][2] * a[1][1]) / det
    a_inv[1][0] = -(a[1][0] * a[2][2] - a[1][2] * a[2][0]) / det
    a_inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det
    a_inv[1][2] = -(a[0][0] * a[1][2] - a[0][2] * a[1][0]) / det
    a_inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det
    a_inv[2][1] = -(a[0][0] * a[2][1] - a[0][1] * a[2][0]) / det
    a_inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det

    return a_inv, det


################# end of inverse 3by3 matrix ######################

################# inverse nbyn matrix ######################
def matinv(a, n):
    ipivot = [0 for j in range(n)]
    pivot = [0 for j in range(n)]
    index = [[0 for i in range(2)] for j in range(n)]
    for i in range(n):
        amax = 0.0
        for j in range(n):
            if ipivot[j] != 1:
                for k in range(n):
                    if ipivot[k] <= 1:
                        if ipivot[k] != 1:
                            if abs(amax) <= abs(a[j][k]):
                                irow = j
                                icolum = k
                                amax = a[j][k]

        ipivot[icolum] += 1
        if irow != icolum:
            for l in range(n):
                swap = a[irow][l]
                a[irow][l] = a[icolum][l]
                a[icolum][l] = swap
        index[i][0] = irow
        index[i][1] = icolum
        pivot[i] = a[icolum][icolum]
        a[icolum][icolum] = 1.0
        for l in range(n):
            a[icolum][l] = a[icolum][l] / pivot[i]
        for l1 in range(n):
            if l1 != icolum:
                t = a[l1][icolum]
                a[l1][icolum] = 0.0
                for l in range(n):
                    a[l1][l] = a[l1][l] - a[icolum][l] * t

    for i in range(n):
        l = n - i - 1
        if index[l][0] != index[l][1]:
            jrow = index[l][0]
            jcolum = index[l][1]
            for k in range(n):
                swap = a[k][jrow]
                a[k][jrow] = a[k][jcolum]
                a[k][jcolum] = swap

    return a


################# end of inverse nbyn matrix ######################

################### apply boundary conditions and solve system equations ##################
def solvesyseq(nbc, gstiff, factor1, f_bc, neq, kbc_n, kbc_d, b_rhs, nbw):
    gstiffm = [[0.0 for i in range(nbw)] for j in range(neq)]

    for ibc in range(nbc):
        jbc = 3 * kbc_n[ibc] + kbc_d[ibc] - 1
        b_rhs[jbc] = gstiff[jbc][jbc] * 1.0e20 * factor1 * f_bc[jbc]
        gstiff[jbc][jbc] = gstiff[jbc][jbc] * 1.0e20

    # ---------- solve with bandwidth-----------------
    # ------ storing total stiffness matrix into neq by nbw matrix ------
    # --------- storing first nbw rows---------
    for i in range(neq - nbw + 1):
        k = -1
        for j in range(i, i + nbw, 1):
            k = k + 1
            gstiffm[i][k] = gstiff[i][j]
    # --------- storing nbw+1 to neq rows---------
    for i in range(neq - nbw + 1, neq):
        for j in range(neq - i):
            gstiffm[i][j] = gstiff[i][j + i]
    # ------ End of storing total stiffness matrix into neq by nbw matrix ------
    # ------- Solve system equations bt Gauss elimination method-------
    for i in range(neq - 1):
        nbk = min(neq - i, nbw)
        for j in range(i + 1, nbk + i, 1):
            j1 = j - i
            c = gstiffm[i][j1] / gstiffm[i][0]
            for l in range(j, nbk + i, 1):
                l1 = l - j
                l2 = l - i
                gstiffm[j][l1] += -c * gstiffm[i][l2]

            b_rhs[j] += -c * b_rhs[i]

    b_rhs[neq - 1] = b_rhs[neq - 1] / gstiffm[neq - 1][0]

    for ii in range(neq - 1):
        i = neq - ii - 2
        nbi = min(neq - i, nbw)
        up = 0.0
        for j in range(1, nbi, 1):
            up += gstiffm[i][j] * b_rhs[i + j - 1 + 1]
        b_rhs[i] = (b_rhs[i] - up) / gstiffm[i][0]

    # ------------ End of solveing with bandwidth ---------

    return b_rhs


################### end of apply boundary conditions and solve system equations ##################

########### post processing ################
def user_print(tnode, nelem, ngauss, gdisp, garray_tau, vol_g):
    stressvol = [0.0 for i in range(6)]
    strainvol = [0.0 for i in range(6)]

    voltot = 0.0

    for ielem in range(nelem):
        for ig in range(ngauss):
            TMP1 = garray_tau[ielem][ig].defg
            TMPt = list(zip(*TMP1))
            TMP2 = [[0.0 for i in range(3)] for j in range(3)]
            TMP2[0][0] = TMP2[1][1] = TMP2[2][2] = 1.0
            TMP3 = [[0.0 for i in range(3)] for j in range(3)]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        TMP3[i][j] += TMPt[i][k] * TMP1[k][j]

            TMP3 = true_strain(TMP3)

            strain = [0.0 for i in range(6)]
            strain[0] = TMP3[0][0];
            strain[1] = TMP3[1][1];
            strain[2] = TMP3[2][2]
            strain[3] = TMP3[0][1];
            strain[4] = TMP3[1][2];
            strain[5] = TMP3[0][2]

            strainvol[0] += strain[0] * vol_g[ielem][ig]
            strainvol[1] += strain[1] * vol_g[ielem][ig]
            strainvol[2] += strain[2] * vol_g[ielem][ig]

    for ielem in range(nelem):
        for ig in range(ngauss):
            stressvol[0] += garray_tau[ielem][ig].cauchy[0][0] * vol_g[ielem][ig]
            stressvol[1] += garray_tau[ielem][ig].cauchy[1][1] * vol_g[ielem][ig]
            stressvol[2] += garray_tau[ielem][ig].cauchy[2][2] * vol_g[ielem][ig]
            voltot += vol_g[ielem][ig]

    stressvol[0] = stressvol[0] / voltot
    stressvol[1] = stressvol[1] / voltot
    stressvol[2] = stressvol[2] / voltot

    strainvol[0] = strainvol[0] / voltot
    strainvol[1] = strainvol[1] / voltot
    strainvol[2] = strainvol[2] / voltot

    return stressvol, strainvol


##################### end of post processing ###################
################ Calculation of logarithmic strain #############
def true_strain(a):
    v = [[0.0 for i in range(3)] for j in range(3)]
    beta = [0.0 for i in range(3)]

    p1 = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2

    if p1 < 1.0e-10:
        for ivec in range(3):
            beta[ivec] = a[ivec][ivec]
            v[ivec][ivec] = 1.0
    elif p1 >= 1.0e-10:

        q = 0.0
        for i in range(3):
            q += a[i][i]
        q = q / 3.0
        Iden = [[0.0 for i in range(3)] for j in range(3)]
        Iden[0][0] = Iden[1][1] = Iden[2][2] = 1.0
        tmp1 = [[0.0 for i in range(3)] for j in range(3)]
        tmp2 = [[0.0 for i in range(3)] for j in range(3)]

        for i in range(3):
            for j in range(3):
                tmp1[i][j] = a[i][j] - q * Iden[i][j]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    tmp2[i][j] += tmp1[i][k] * tmp1[k][j]
                tmp2[i][j] = tmp2[i][j] / 6.0
        p = 0.0
        for i in range(3):
            p += tmp2[i][i]
        p = math.sqrt(p)

        b = [[0.0 for i in range(3)] for j in range(3)]
        for i in range(3):
            for j in range(3):
                b[i][j] = tmp1[i][j] / p

        det_b = b[0][0] * b[1][1] * b[2][2] - b[0][0] * b[1][2] * b[2][1] - b[1][0] * b[0][1] \
                * b[2][2] + b[1][0] * b[0][2] * b[2][1] + b[2][0] * b[0][1] * b[1][2] - b[2][0] \
                * b[0][2] * b[1][1]
        r = det_b / 2.0
        if r <= -1.0:
            phi = math.pi / 3.0
        elif r >= 1.0:
            phi = 0.0
        elif r > -1.0 and r < 1.0:
            phi = math.acos(r) / 3.0

        beta[0] = q + 2.0 * p * math.cos(phi)
        beta[1] = q + 2.0 * p * math.cos(phi + (2.0 * math.pi / 3.0))
        beta[2] = 3.0 * q - beta[0] - beta[1]

        tmp3 = [[0.0 for i in range(3)] for j in range(3)]

        for ivec in range(3):
            for i in range(3):
                for j in range(3):
                    tmp3[i][j] = a[i][j] - beta[ivec] * Iden[i][j]

            if tmp3[2][2] != 0.0:
                alpha1 = 1.0
                alpha21 = -tmp3[1][0] + (tmp3[1][2] * tmp3[2][0] / tmp3[2][2])
                alpha22 = tmp3[1][1] - (tmp3[1][2] * tmp3[2][1] / tmp3[2][2])
                alpha2 = alpha21 / alpha22
                alpha3 = (-tmp3[2][0] / tmp3[2][2]) - (tmp3[2][1] / tmp3[2][2]) * alpha2
            if tmp3[2][2] == 0.0 and tmp3[2][1] != 0.0:
                alpha1 = 1.0
                alpha2 = -tmp3[2][0] / tmp3[2][1]
                if tmp3[1][2] != 0.0:
                    alpha3 = (tmp3[1][0] - tmp3[1][1] * alpha2) / tmp3[1][2]
                elif tmp3[1][2] == 0.0 and tmp3[0][2] != 0.0:
                    alpha3 = (tmp3[0][0] - tmp3[0][1] * alpha2) / tmp3[1][2]
                elif tmp3[1][2] == 0.0 and tmp3[0][2] == 0.0:
                    alpha3 = 0.0

            if tmp3[2][2] == 0.0 and tmp3[2][1] == 0.0 and tmp3[2][1] != 0.0:
                alpha1 = 0.0
                if tmp3[1][2] != 0.0:
                    alpha2 = 1.0
                    alpha3 = -tmp3[1][1] / tmp3[1][2]
                elif tmp3[1][2] == 0.0 and tmp3[0][2] != 0.0:
                    alpha2 = 0.0
                    alpha3 = 1.0

            alpha = math.sqrt(alpha1 ** 2 + alpha2 ** 2 + alpha3 ** 2)
            alpha1 = alpha1 / alpha;
            alpha2 = alpha2 / alpha;
            alpha3 = alpha3 / alpha
            v[ivec][0] = alpha1;
            v[ivec][1] = alpha2;
            v[ivec][2] = alpha3

    strain = [[0.0 for i in range(3)] for j in range(3)]

    tmpv1 = [[0.0 for i in range(3)] for j in range(3)]
    tmpv2 = [[0.0 for i in range(3)] for j in range(3)]
    tmpv3 = [[0.0 for i in range(3)] for j in range(3)]

    for i in range(3):
        for j in range(3):
            tmpv1[i][j] = v[0][i] * v[0][j]

    for i in range(3):
        for j in range(3):
            tmpv2[i][j] = v[1][i] * v[1][j]

    for i in range(3):
        for j in range(3):
            tmpv3[i][j] = v[2][i] * v[2][j]
    for i in range(3):
        for j in range(3):
            strain[i][j] = 0.0
            strain[i][j] = 0.5 * math.log(beta[0]) * tmpv1[i][j] + \
                           0.5 * math.log(beta[1]) * tmpv2[i][j] + \
                           0.5 * math.log(beta[2]) * tmpv3[i][j]

    return strain


################ end of calculation of logarithmic strain ######


#################### start off classes ###########


class Node:
    def __init__(self, inode, coorx, coory, coorz):
        self.inode = inode
        self.coorx = coorx
        self.coory = coory
        self.coorz = coorz

    def __repr__(self):
        return "(%d,%f,%f,%f)" % (self.inode, self.coorx, self.coory, self.coorz)


class Element:
    def __init__(self, idx, nodelist=[]):
        self.index = idx
        self.nodes = nodelist[:]

    def __repr__(self):
        return "Element(%d,%s)" % (self.index, self.nodes)


####################### Flow rule #########################
class update_statev:
    def __init__(self, tau_alpha, res_ssd, dgam, dgam_dta, prop):
        self.tau_alpha = tau_alpha
        self.res_ssd = res_ssd
        self.dgam = dgam
        self.dgam_dta = dgam_dta
        self.prop = prop

    def RES(self):

        const_w1 = self.prop[4];
        const_w2 = self.prop[5]
        const_ss = self.prop[6]
        const_a = self.prop[7]
        const_h0 = self.prop[8];
        const_m = self.prop[9]
        g0dot = self.prop[10]
        dt = self.prop[11]

        n_slip = len(self.tau_alpha)
        res = [0.0 for i in range(n_slip)]
        res_t = [0.0 for i in range(n_slip)]
        dgamma = [0.0 for i in range(n_slip)]
        dgam_dtau = [0.0 for i in range(n_slip)]

        #       for i in range(n_slip):
        res0 = self.res_ssd
        dgamma = self.dgam
        dgam_dtau = self.dgam_dta

        for k in range(n_slip):
            res[k] = 0.0
            for i in range(n_slip):
                if i == k:
                    const_qab = const_w1
                elif i != k:
                    const_qab = const_w2

                ratio_res = 1.0 - (res0[i] / const_ss)

                res[k] += const_qab * const_h0 * abs(dgamma[i]) * ((ratio_res) ** const_a)

        for k in range(n_slip):
            res[k] += res0[k]

        for k in range(n_slip):
            if res[k] >= 1.0:

                ratio_alpha = self.tau_alpha[k] / res[k]

                if self.tau_alpha[k] >= 0.0:
                    const_sign = 1.0
                elif self.tau_alpha[k] < 0.0:
                    const_sign = -1.0

                m_inv = 1.0 / const_m
                dgamma[k] = dt * const_sign * g0dot * ((abs(ratio_alpha)) ** m_inv)

                ######	Calculation of dgamma_dtau (The sgn(Tau_alpha) is multiplied with another sgn(Tau_alpha) from the derivative)
                res_inv = 1.0 / res[k]

                dgam_dtau[k] = dt * res_inv * g0dot * m_inv * ((abs(ratio_alpha)) ** (m_inv - 1.0))

            elif res[k] < 1.0:

                dgamma[k] = 0.0
                dgam_dtau[k] = 0.0

        return dgamma, dgam_dtau, res


################ User Material ###########

class UMAT:
    def __init__(self, S_trial, C_mat, Fp, C_alpha, schmid, F_t, F_tau, res, dgam, dgam_dta, props, S_star0):
        self.S_trial = S_trial
        self.C_mat = C_mat
        self.Fp = Fp
        self.C_alpha = C_alpha
        self.schmid = schmid
        self.F_t = F_t
        self.F_tau = F_tau
        self.res = res
        self.dgam = dgam
        self.dgam_dta = dgam_dta
        self.props = props
        self.S_star0 = S_star0

    #        self.tau_alpha = [0.0 for k in range(len(self.schmid))]

    # ------------ Forth order Kronecker Delta ---------------------
    def delta_kron4(self):
        delta_kron = [[0. for ii in range(3)] for jj in range(3)]
        delta_kron[0][0] = delta_kron[1][1] = delta_kron[2][2] = 1.0

        delta_kron4d = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        delta_kron4d[i][j][k][l] += delta_kron[i][k] * delta_kron[j][l] + delta_kron[i][l] * \
                                                    delta_kron[j][k]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        delta_kron4d[i][j][k][l] = 0.5 * delta_kron4d[i][j][k][l]

        return delta_kron4d

    # ---------------------------------------------------------------

    # ------------------ Iteration process for time integration of Fp --------

    def itr(self):
        n_slip = len(self.schmid)

        tau_alpha = [0.0 for k in range(n_slip)]
        tau_alpha = resolvedshear(n_slip, self.S_trial, self.schmid)

        res_ssd = self.res
        dgam = self.dgam
        dgam_dta = self.dgam_dta
        dgam = [0.0 for k in range(n_slip)]
        dgam_dta = [0.0 for k in range(n_slip)]

        # ---------------------- Update state variables ----------------
        prop = self.props
        s1 = update_statev(tau_alpha, res_ssd, dgam, dgam_dta, prop)
        u1 = s1.RES()
        dgam = u1[0]
        dgam_dta = u1[1]
        #        res_ssd = u1[2]

        TMP1 = [[0.0 for it in range(3)] for jt in range(3)]
        for it in range(3):
            for jt in range(3):
                for kt in range(n_slip):
                    TMP1[it][jt] += dgam[kt] * self.C_alpha[kt][it][jt]

        # -------------------- Calculation of first value of 2nd PKS-----------------
        S_star = [[0.0 for it in range(3)] for jt in range(3)]
        for it in range(3):
            for jt in range(3):
                S_star[it][jt] = self.S_trial[it][jt] - TMP1[it][jt]
        #        S_star = self.S_star0

        #        print "Start of iteration loop for time integration"
        # ----------------- iteration loop for S_star -------------------
        niter = 20
        iitr = 0
        ratio_norm = 1.0e10
        while (iitr <= 20 and abs(ratio_norm) >= 0.001):

            # ------------ Get the resolved shear stress and update state variables with S_star-----------------
            tau_alpha = [0.0 for k in range(n_slip)]
            tau_alpha = resolvedshear(n_slip, S_star, self.schmid)

            s2 = update_statev(tau_alpha, res_ssd, dgam, dgam_dta, prop)
            u2 = s2.RES()
            dgam = u2[0]
            dgam_dta = u2[1]
            #            res_ssd = u2[2]

            # ---------
            GT_mat = [[[0. for i in range(3)] for j in range(3)] for k in range(n_slip)]

            for k in range(n_slip):
                for i in range(3):
                    for j in range(3):
                        GT_mat[k][i][j] = 0.5 * (self.schmid[k][i][j] + self.schmid[k][j][i]) * dgam_dta[k]
            self._GT_mat_chache = GT_mat
            # -----------
            # ----------
            RJ_mat = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]
            for i in range(3):
                for j in range(3):
                    for m in range(3):
                        for n in range(3):
                            for k in range(n_slip):
                                RJ_mat[i][j][m][n] += self.C_alpha[k][i][j] * GT_mat[k][m][n]

            # -------------
            delta4 = UMAT.delta_kron4(self)
            for i in range(3):
                for j in range(3):
                    for m in range(3):
                        for n in range(3):
                            RJ_mat[i][j][m][n] += delta4[i][j][m][n]

            RJ_reduced = reduce_mat(RJ_mat)
            # --------------
            # ------------

            RJ_reduced = matinv(RJ_reduced, 6)

            self._RJ_reduced_cache = RJ_reduced

            # ---------------
            # -------------
            GN_vec = calc_GNvec(n_slip, S_star, self.S_trial, dgam, self.C_alpha)
            # ----------------
            # -----------------

            S_vec = reduce_vec(S_star)

            TMP_RG = [0.0 for i in range(6)]
            TMP_vec = [0.0 for i in range(6)]

            for i in range(6):
                for j in range(6):
                    TMP_RG[i] += RJ_reduced[i][j] * GN_vec[j]

            for i in range(6):
                TMP_vec[i] = S_vec[i]

            for i in range(6):
                S_vec[i] -= TMP_RG[i]

            S_star = inflate_vec(S_vec)

            dot_product_S_vec = 0.0
            for i in range(6):
                dot_product_S_vec += S_vec[i] * S_vec[i]

            rnorm_s_vec = math.sqrt(dot_product_S_vec)

            dot_product_TMP_vec = 0.0
            for i in range(6):
                dot_product_TMP_vec += TMP_vec[i] * TMP_vec[i]

            rnorm_tmp_vec = math.sqrt(dot_product_TMP_vec)

            if abs(rnorm_tmp_vec) < 1.0:
                ratio_norm = 0.0
            elif abs(rnorm_tmp_vec) >= 1.0:
                ratio_norm = (rnorm_s_vec - rnorm_tmp_vec) / rnorm_tmp_vec
            # --------------------

            iitr += 1

        # ------------ Get the resolved shear stress and update state variables with the updated S_star------------
        tau_alpha = [0.0 for k in range(n_slip)]
        tau_alpha = resolvedshear(n_slip, S_star, self.schmid)

        s3 = update_statev(tau_alpha, res_ssd, dgam, dgam_dta, prop)
        u3 = s3.RES()
        dgam = u3[0]
        dgam_dta = u3[1]
        res_ssd = u3[2]

        self._dgam_cache = dgam
        self._S_star_cache = S_star

        # ---------- Calculation of Velocity Gradient --------------

        Lp = [[0.0 for i in range(3)] for j in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(n_slip):
                    Lp[i][j] += dgam[k] * self.schmid[k][i][j]
        # ----------- Calculation of Fp-tau ---------------------
        Iden = [[0. for i in range(3)] for j in range(3)]
        Fp_tau = [[0. for i in range(3)] for j in range(3)]

        Iden[0][0] = Iden[1][1] = Iden[2][2] = 1.0
        for i in range(3):
            for j in range(3):
                Lp[i][j] += Iden[i][j]

        Fp_tau = matmul(Lp, self.Fp)

        # Here
        det_Fp_tau = (Fp_tau[0][0] * Fp_tau[1][1] * Fp_tau[2][2] \
                      - Fp_tau[0][0] * Fp_tau[1][2] * Fp_tau[2][1] - Fp_tau[1][0] * Fp_tau[0][1] * Fp_tau[2][2] \
                      + Fp_tau[1][0] * Fp_tau[0][2] * Fp_tau[2][1] + Fp_tau[2][0] * Fp_tau[0][1] * Fp_tau[1][2] \
                      - Fp_tau[2][0] * Fp_tau[0][2] * Fp_tau[1][1])

        oby3 = 1.0 / 3.0
        for i in range(3):
            for j in range(3):
                Fp_tau[i][j] = Fp_tau[i][j] / ((det_Fp_tau) ** oby3)

        # ----------- Calculation of the new Cauchy Stress ------------

        Fe_tau = [[0.0 for i in range(3)] for j in range(3)]
        Fp_tauI = [[0.0 for i in range(3)] for j in range(3)]
        FptauD = matinv3(Fp_tau)
        Fp_tauI = FptauD[0]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    Fe_tau[i][j] += self.F_tau[i][k] * Fp_tauI[k][j]

        TPMc = [[0.0 for i in range(3)] for j in range(3)]
        TPMc = matmul(S_star, zip(*Fe_tau))

        Cauchy = [[0.0 for i in range(3)] for j in range(3)]
        Cauchy = matmul(Fe_tau, TPMc)

        det_F_tau = (self.F_tau[0][0] * self.F_tau[1][1] * self.F_tau[2][2] \
                     - self.F_tau[0][0] * self.F_tau[1][2] * self.F_tau[2][1] - self.F_tau[1][0] * self.F_tau[0][1] *
                     self.F_tau[2][2] \
                     + self.F_tau[1][0] * self.F_tau[0][2] * self.F_tau[2][1] + self.F_tau[2][0] * self.F_tau[0][1] *
                     self.F_tau[1][2] \
                     - self.F_tau[2][0] * self.F_tau[0][2] * self.F_tau[1][1])

        for i in range(3):
            for j in range(3):
                Cauchy[i][j] = Cauchy[i][j] / det_F_tau

        Iden = [[0. for i in range(3)] for j in range(3)]
        TMPe = [[0. for i in range(3)] for j in range(3)]
        Strain_el = [[0. for i in range(3)] for j in range(3)]
        Iden[0][0] = Iden[1][1] = Iden[2][2] = 1.0

        TMPe = matmul(zip(*Fe_tau), Fe_tau)
        for i in range(3):
            for j in range(3):
                Strain_el[i][j] = 0.5 * (TMPe[i][j] - Iden[i][j])

        self._Fe_tau_cache = Fe_tau
        return Fe_tau, Fp_tau, Cauchy, res_ssd, tau_alpha, dgam, dgam_dta, S_star

    # ------------------ End of Iteration method for time integration of Fp --------
    # --------------------------------------------------------------

    # ---------------------- Polar Decomposition of Rlative Gradient Deformation -------------------
    def polardecomp(self):

        # -------------- Calculation of relative deformation gradient F_REL = F_tau*(F_t)^-1 -----------
        F_REL = [[0.0 for i in range(3)] for j in range(3)]

        F_tI = [[0. for i in range(3)] for j in range(3)]

        F_tID = matinv3(self.F_t)
        F_tI = F_tID[0]

        for i in range(3):
            for j in range(3):
                F_REL[i][j] = 0.0
                for k in range(3):
                    F_REL[i][j] += self.F_tau[i][k] * F_tI[k][j]

        # ----------------------- square root of a positive matrix U=sqrt(C) ------------

        # ----------------------- square root of a positive matrix U=sqrt(C) ------------

        R = [[0. for i in range(3)] for j in range(3)]
        C = [[0. for i in range(3)] for j in range(3)]
        Csquare = [[0. for i in range(3)] for j in range(3)]
        Iden = [[0. for i in range(3)] for j in range(3)]
        U = [[0. for i in range(3)] for j in range(3)]
        invU = [[0. for i in range(3)] for j in range(3)]

        Iden[0][0] = Iden[1][1] = Iden[2][2] = 1.0

        F_REL_T = list(zip(*F_REL))
        # ----------------------- C = FTF --------------------------------------
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C[i][j] += F_REL_T[i][k] * F_REL[k][j]

        # ---------------------- C^2 ------------------------------------------
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    Csquare[i][j] += C[i][k] * C[k][j]

        o3 = 1.0 / 3.0;
        root3 = math.sqrt(3.0)
        c1212 = C[0][1] * C[0][1]
        c1313 = C[0][2] * C[0][2]
        c2323 = C[1][2] * C[1][2]
        c2313 = C[1][2] * C[0][2]
        c1223 = C[0][1] * C[1][2]
        c1213 = C[0][1] * C[0][2]
        s11 = C[1][1] * C[2][2] - c2323
        ui1 = o3 * (C[0][0] + C[1][1] + C[2][2])
        ui2 = s11 + C[0][0] * C[1][1] + C[2][2] * C[0][0] - c1212 - c1313
        ui3 = C[0][0] * s11 + C[0][1] * (c2313 - C[0][1] * C[2][2]) + C[0][2] * (c1223 - C[1][1] * C[0][2])
        ui1s = ui1 * ui1
        q = math.sqrt(-min((o3 * ui2 - ui1s), 0.0))
        r = 0.5 * (ui3 - ui1 * ui2) + ui1 * ui1s
        xmod = q * q * q

        sign = 0.0
        if (xmod - 1.0e30 > 0.0):
            sign = 1.0
        else:
            sign = -1.0

        scl1 = 0.5 + 0.5 * sign

        if (xmod - abs(r) > 0.0):
            sign = 1.0
        else:
            sign = -1.0

        scl2 = 0.5 + 0.5 * sign
        scl0 = fmin(scl1, scl2)
        scl1 = 1.0 - scl0

        xmodscl1 = 0.0
        if (scl1 == 0):
            xmodscl1 = xmod
        else:
            xmodscl1 = xmod + scl1

        sdetm = math.acos(r / (xmodscl1)) * o3

        q = scl0 * q
        ct3 = q * math.cos(sdetm)
        st3 = q * root3 * math.sin(sdetm)
        sdetm = scl1 * math.sqrt(max(0.0, r))
        aa = 2.0 * (ct3 + sdetm) + ui1
        bb = -ct3 + st3 - sdetm + ui1
        cc = -ct3 - st3 - sdetm + ui1

        # --------------------- Eigenvalues of U -----------------------------

        lamda1 = math.sqrt(max(aa, 0.0))
        lamda2 = math.sqrt(max(bb, 0.0))
        lamda3 = math.sqrt(max(cc, 0.0))

        # --------------------- Invarients of U ---------------------------

        Iu = lamda1 + lamda2 + lamda3;
        IIu = lamda1 * lamda2 + lamda1 * lamda3 + lamda2 * lamda3;
        IIIu = lamda1 * lamda2 * lamda3;

        # --------------------- U and Inverse of U ------------------------

        for i in range(3):
            for j in range(3):
                U[i][j] = Iu * IIIu * Iden[i][j] + (Iu ** 2 - IIu) * C[i][j] - Csquare[i][j]
                U[i][j] = U[i][j] / (Iu * IIu - IIIu)

        for i in range(3):
            for j in range(3):
                invU[i][j] = (IIu * Iden[i][j] - Iu * U[i][j] + C[i][j]) / IIIu;

        # ----------------------- R = FU^-1 -----------------------------

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    R[i][j] += F_REL[i][k] * invU[k][j]

        # ---------------------- End of Polar Decomposition ---------------

        return U, R

    # ---------------- Calculation of elasto plastic modulus or elastic Jacobian ---------

    def W_mat(self):

        # ---------Elastic stretch and rotation------------
        Poldecomp = UMAT.polardecomp(self)
        Strh_el = Poldecomp[0]
        Rot_el = Poldecomp[1]
        # ------------------------------------------------

        Fp_I = matinv3(self.Fp)[0]

        Fe_t = matmul(self.F_t, Fp_I)

        Fe_tau = self._Fe_tau_cache
        # ------ Calculation of L_mat-------
        L_mat = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        L_mat[i][j][k][l] = 0.0
                        for m in range(3):
                            L_mat[i][j][k][l] += Fe_t[k][i] * Strh_el[l][m] * Fe_t[m][j] + Fe_t[m][i] * Strh_el[m][k] * \
                                                 Fe_t[l][j]

        # --------- Calculation of D_mat ---------
        D_mat = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        D_mat[i][j][k][l] = 0.0
                        for m in range(3):
                            for n in range(3):
                                D_mat[i][j][k][l] += 0.5 * self.C_mat[i][j][m][n] * L_mat[m][n][k][l]

        # -------------- Calculation of G_alpha----------------------
        G_alpha = [[[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)] for alpha in
                   range(len(self.schmid))]

        for alpha in range(len(self.schmid)):
            for m in range(3):
                for n in range(3):
                    for k in range(3):
                        for l in range(3):
                            G_alpha[alpha][m][n][k][l] = 0.0
                            for p in range(3):
                                G_alpha[alpha][m][n][k][l] += L_mat[m][p][k][l] * self.schmid[alpha][p][n] + \
                                                              self.schmid[alpha][m][p] * L_mat[p][n][k][l]
        # --------- Calculation of J_alpha ---------
        J_alpha = [[[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)] for alpha in
                   range(len(self.schmid))]

        for alpha in range(len(self.schmid)):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            J_alpha[alpha][i][j][k][l] = 0.0
                            for m in range(3):
                                for n in range(3):
                                    J_alpha[alpha][i][j][k][l] += 0.5 * self.C_mat[i][j][m][n] * \
                                                                  G_alpha[alpha][m][n][k][l]

        # ------------- Calculation of Q_mat -------------------
        RJ_reduced = self._RJ_reduced_cache
        K_inv = inflate_ten(RJ_reduced)

        dgam = self._dgam_cache

        TMP_4d = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]
        for m in range(3):
            for n in range(3):
                for k in range(3):
                    for l in range(3):
                        TMP_4d[m][n][k][l] = 0.0
                        for alpha in range(len(self.schmid)):
                            TMP_4d[m][n][k][l] += dgam[alpha] * J_alpha[alpha][m][n][k][l]

        Q_mat = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        Q_mat[i][j][k][l] = 0.0
                        for m in range(3):
                            for n in range(3):
                                Q_mat[i][j][k][l] += K_inv[i][j][m][n] * (D_mat[m][n][k][l] - TMP_4d[m][n][k][l])

        # -------------- Calculation of R_alpha -----------------
        R_alpha = [[[0. for i in range(3)] for j in range(3)] for k in range(len(self.schmid))]
        GT_mat = self._GT_mat_chache
        for alpha in range(len(self.schmid)):
            for i in range(3):
                for j in range(3):
                    R_alpha[alpha][i][j] = 0.0
                    for k in range(3):
                        for l in range(3):
                            R_alpha[alpha][i][j] += GT_mat[alpha][k][l] * Q_mat[k][l][i][j]
        # ------------- Calculation of S_mat---------------------
        TMP_4d = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]
        for k in range(3):
            for l in range(3):
                for p in range(3):
                    for j in range(3):
                        TMP_4d[k][l][p][j] = 0.0
                        for alpha in range(len(self.schmid)):
                            TMP_4d[k][l][p][j] += R_alpha[alpha][k][l] * self.schmid[alpha][p][j]

        TMP_2d = [[0. for i in range(3)] for j in range(3)]
        for p in range(3):
            for j in range(3):
                TMP_2d[p][j] = 0.0
                for alpha in range(len(self.schmid)):
                    TMP_2d[p][j] += dgam[alpha] * self.schmid[alpha][p][j]

        S_mat = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        S_mat[i][j][k][l] = Rot_el[i][k] * Fe_tau[l][j]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for p in range(3):
                            S_mat[i][j][k][l] -= Rot_el[i][k] * Fe_tau[l][p] * TMP_2d[p][j]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            for n in range(3):
                                for p in range(3):
                                    S_mat[i][j][k][l] -= Rot_el[i][m] * Strh_el[m][n] * Fe_tau[n][p] * TMP_4d[k][l][p][
                                        j]

        # ------------------ Calculation of inverse of Fe ----------------

        Fe_tau_inv = [[0.0 for i in range(3)] for j in range(3)]
        Fe_tauID = matinv3(Fe_tau)
        Fe_tau_inv = Fe_tauID[0];
        det_Fe_tau = Fe_tauID[1]

        S_star = self._S_star_cache

        W_mat = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            for n in range(3):
                                W_mat[i][j][k][l] += S_mat[i][m][k][l] * S_star[m][n] * Fe_tau[j][n]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            for n in range(3):
                                W_mat[i][j][k][l] += Fe_tau[i][m] * Q_mat[m][n][k][l] * Fe_tau[j][n]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            for n in range(3):
                                W_mat[i][j][k][l] += Fe_tau[i][m] * S_star[m][n] * S_mat[j][n][k][l]
        TMP_sf = [[0.0 for i in range(3)] for j in range(3)]
        for k in range(3):
            for l in range(3):
                for p in range(3):
                    for q in range(3):
                        TMP_sf[k][l] += S_mat[p][q][k][l] * Fe_tau_inv[q][p]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            for n in range(3):
                                W_mat[i][j][k][l] -= Fe_tau[i][m] * S_star[m][n] * Fe_tau[j][n] * TMP_sf[k][l]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        W_mat[i][j][k][l] = W_mat[i][j][k][l] / det_Fe_tau

        Dep = reduce_wmat(W_mat)

        return Dep


class array:
    def __init__(self, ielem, igauss, cauchy, defg, pdefg, dep, res_ssd, dgam, dgam_dta, S_star):
        self.ielem = ielem
        self.igauss = igauss
        self.cauchy = cauchy

        self.defg = defg
        self.pdefg = pdefg
        self.dep = dep
        self.res_ssd = res_ssd
        self.dgam = dgam
        self.dgam_dta = dgam_dta
        self.S_star = S_star

    def __repr__(self):
        return "(%d,%d,%s,%s,%s,%s,%s,%s)" % (
        self.ielem, self.igauss, self.cauchy, self.defg, self.pdefg, self.dep, self.res_ssd, self.dgam, self.dgam_dta,
        self.S_star)


class darray:
    def __init__(self, inode, ux, uy, uz):
        self.inode = inode
        self.ux = ux
        self.uy = uy
        self.uz = uz

    def __repr__(self):
        return "(%d,%f,%f,%f)" % (self.inode, self.ux, self.uy, self.uz)


class Mesh:
    def __init__(self, nnode, tnode, coors, nelem, elements):  # xelements=2,yelements=2,zelements=2):
        self.nodelist = []
        self.ellist = []

        for inode in range(tnode):
            node = Node(coors[inode][0], coors[inode][1], coors[inode][2], coors[inode][3])
            self.nodelist.append(node)

        element_index = 0
        for iele in range(nelem):
            nodes = [self.nodelist[elements[iele][j + 2]] for j in range(nnode)]
            self.ellist.append(Element(element_index, nodes))
            element_index += 1

    ############ starting element loop #####################
    def solve(self, nbc, bcs, angle, props, tstep, nconv, n_slip):

        ##### Calculate the global load vector(here it is displacement control only) #####
        ### nbc=total bcs, kbc[node number][direction x=1,y=2,z=3
        nstep = tstep[0];
        dt = tstep[1];
        strain_rate0 = tstep[2]

        neq = 3 * len(self.nodelist)

        kbc_n = [0 for i in range(nbc)]
        kbc_d = [0 for i in range(nbc)]
        kbc_v = [0 for i in range(nbc)]

        for ibc in range(nbc):
            kbc_n[ibc] = bcs[ibc][0] - 1
            kbc_d[ibc] = bcs[ibc][1]
            kbc_v[ibc] = bcs[ibc][2]

        f_bc = [0.0 for i in range(neq)]

        for ibc in range(nbc):
            f_bc[3 * kbc_n[ibc] + kbc_d[ibc] - 1] = kbc_v[ibc]

        c11 = props[0];
        c12 = props[1];
        c44 = props[2]
        res0_ssd = props[3]

        nbw = 0

        ##### end of load vector #######
        ####### initialisation ######
        crystal_type = 'fcc'

        XYZ = [[0.0 for i in range(3)] for j in range(8)]
        xyz = [[0.0 for i in range(3)] for j in range(8)]

        garray_tau = []
        vol_g = []

        gstiff = [[0.0 for i in range(neq)] for j in range(neq)]
        b_rhs = [0.0 for i in range(neq)]
        gf_nod = [0.0 for i in range(neq)]
        unb_nod = [0.0 for i in range(neq)]

        for ielem in self.ellist:

            nnode = 8

            lnode = [ielem.nodes[i].inode for i in range(nnode)]

            nb1 = max(lnode)
            nb2 = min(lnode)
            nb = 3 * (nb1 - nb2 + 1)
            if nb > nbw:
                nbw = nb

            array_tau = []

            qrot = [[0. for ii in range(3)] for jj in range(3)]
            qrot[0][0] = 1.0;
            qrot[1][1] = 1.0;
            qrot[2][2] = 1.0

            angle_el = angle[ielem.index]
            qrot = euler_rot(angle_el, qrot)

            res_ssd = [res0_ssd for ir in range(n_slip)]
            dgam = [0.0 for ir in range(n_slip)]
            dgam_dta = [0.0 for ir in range(n_slip)]
            S_star0 = [[0.0 for ir in range(3)] for jr in range(3)]

            for i in range(nnode):
                XYZ[i][0] = ielem.nodes[i].coorx
                XYZ[i][1] = ielem.nodes[i].coory
                XYZ[i][2] = ielem.nodes[i].coorz

                for j in range(3):
                    xyz[i][j] = XYZ[i][j]

            schmid = calc_schmid(crystal_type, n_slip, qrot)

            C_mat = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]

            C_mat = cmat(c11, c12, c44, qrot)

            gaussw = pgauss()

            ngauss = 8

            strs = [[0.0 for i in range(6)] for j in range(ngauss)]

            # ---------------------- Start Calculation for Gauss Points

            for ig in range(ngauss):
                # ---------------------- F_t = F_tau previous -------------

                F_t = [[0. for ii in range(3)] for jj in range(3)]
                F_t[0][0] = F_t[1][1] = F_t[2][2] = 1.0

                # ------- Get the Deformation Gradient at time = tau for gauss points for the given element

                SHP = SHP3D(gaussw[0][ig], gaussw[1][ig], gaussw[2][ig], XYZ)

                F_tau = calc_F(xyz, SHP[0])

                ########### Initialization of Plastic Defomation Gradient ######
                Fp = [[0. for ii in range(3)] for jj in range(3)]
                Fp[0][0] = Fp[1][1] = Fp[2][2] = 1.0

                # ------------------- Calculate matrix [A]
                A_mat = calc_A(F_tau, Fp)

                # ------------------  Calculate matrix [B_alpha]
                B_alpha = calc_B(n_slip, A_mat, schmid)

                # ----------------- Calculate the trial elastic stress S_trial
                S_trial = stress_trial(C_mat, A_mat)

                # ---------------- Calculate the matrix C_alpha for each slip system
                C_alpha = calc_C_alpha(n_slip, C_mat, B_alpha)

                ######################## Enter the User Material ###################

                U = UMAT(S_trial, C_mat, Fp, C_alpha, schmid, F_t, F_tau, res_ssd, dgam, dgam_dta, props, S_star0)

                U1 = U.itr()

                U2 = U.W_mat()

                strs[ig][0] = U1[2][0][0];
                strs[ig][1] = U1[2][1][1];
                strs[ig][2] = U1[2][2][2]
                strs[ig][3] = U1[2][0][1];
                strs[ig][4] = U1[2][1][2];
                strs[ig][5] = U1[2][0][2]

                #####################     U1[0]=Fe_tau, U1[1]=Fp_tau, U1[2]=Cauchy, U1[3]=res_ssd, U1[5]=dgam , U1[6]=dgam_dta
                mg = array(ielem.index, ig, U1[2], F_tau, U1[1], U2, U1[3], U1[5], U1[6], U1[7])

                array_tau.append(mg)
            ########### End of gauss point loop ##############

            ############### Element reaction forces or internal forces ################

            InF = element_re(ngauss, nnode, xyz, strs)

            gf_nod = asembl_vec(gf_nod, InF, lnode)

            ############## End of Element reaction forces or internal forces ################

            garray_tau.append(array_tau)

            ############# Calculate the stiffness matrix ###############

            dep = [garray_tau[ielem.index][ig].dep for ig in range(ngauss)]

            ev = STIF3D(ngauss, nnode, xyz, strs, dep)
            estiff = ev[0]
            vol_g.append(ev[1])

            ######## asembel total stiffness matrix ##############
            for i in range(nnode):
                for j in range(nnode):
                    for k in range(3):
                        for l in range(3):
                            gstiff[3 * ielem.nodes[i].inode + k][3 * ielem.nodes[j].inode + l] += estiff[3 * i + k][
                                3 * j + l]

        ######## end of asembel total stiffness matrix ##################

        tnode = len(self.nodelist)
        nelem = len(self.ellist)

        strsvol = []
        strnvol = []

        gdisp = [[0.0 for i in range(3)] for j in range(tnode)]

        time = 0.0

        facload = [dt for i in range(nstep)]
        strain_rate = [strain_rate0 for i in range(nstep)]

        Cauchy_stress = []
        Fp_tau = []
        Ftau = []

        for istep in range(nstep):

            for ibc in range(nbc):
                f_bc[3 * kbc_n[ibc] + kbc_d[ibc] - 1] = kbc_v[ibc]

            factor1 = facload[istep] * strain_rate[istep]
            dt = facload[istep];
            time = time + dt

            # ----------------- apply boundary conditions and solve system equations -----------
            #            b_rhs = solvesmallmatrix(nbc,gstiff,factor1,f_bc,neq,kbc_n,kbc_d,b_rhs)
            b_rhs = solvesyseq(nbc, gstiff, factor1, f_bc, neq, kbc_n, kbc_d, b_rhs, nbw)
            # ----------------------------------------------------------------------------------

            for i in range(tnode):
                for j in range(3):
                    gdisp[i][j] += b_rhs[3 * i + j]

            iconv = 0
            ratio_norm = 1.0e10
            while (iconv <= nconv and abs(ratio_norm) >= 1.0e-9):

                if iconv > 0:

                    for i in range(neq):
                        b_rhs[i] = unb_nod[i]

                    f_bc = [0.0 for i in range(neq)]
                    #            b_rhs = solvesmallmatrix(nbc,gstiff,factor1,f_bc,neq,kbc_n,kbc_d,b_rhs)
                    b_rhs = solvesyseq(nbc, gstiff, factor1, f_bc, neq, kbc_n, kbc_d, b_rhs, nbw)

                    for i in range(tnode):
                        for j in range(3):
                            gdisp[i][j] += b_rhs[3 * i + j]

                gf_nod = [0.0 for i in range(neq)]
                unb_nod = [0.0 for i in range(neq)]
                gstiff = [[0.0 for i in range(neq)] for j in range(neq)]

                garray_t = garray_tau
                garray_tau = []
                vol_g = []

                # ----------------------- start of loop element --------------------------
                for ielem in self.ellist:

                    array_tau = []
                    disp = [0.0 for i in range(24)]
                    XYZ = [[0.0 for i in range(3)] for j in range(8)]
                    xyz = [[0.0 for i in range(3)] for j in range(8)]

                    qrot = [[0. for ii in range(3)] for jj in range(3)]
                    qrot[0][0] = 1.0;
                    qrot[1][1] = 1.0;
                    qrot[2][2] = 1.0

                    angle_el = angle[ielem.index]

                    qrot = euler_rot(angle_el, qrot)

                    # ------------ Coordinates in the refrence configuration
                    # ------------ Assign element-node relationship
                    for i in range(nnode):
                        XYZ[i][0] = ielem.nodes[i].coorx
                        XYZ[i][1] = ielem.nodes[i].coory
                        XYZ[i][2] = ielem.nodes[i].coorz

                        disp[i * 3 + 0] = gdisp[ielem.nodes[i].inode][0]
                        disp[i * 3 + 1] = gdisp[ielem.nodes[i].inode][1]
                        disp[i * 3 + 2] = gdisp[ielem.nodes[i].inode][2]

                        for j in range(3):
                            xyz[i][j] = XYZ[i][j] + disp[i * 3 + j]

                    schmid = calc_schmid(crystal_type, n_slip, qrot)

                    C_mat = [[[[0. for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]

                    C_mat = cmat(c11, c12, c44, qrot)

                    strs = [[0.0 for i in range(6)] for j in range(ngauss)]

                    # ---------------------- Start Calculation for Gauss Points

                    # ------- Get the Deformation Gradient at time = tau for gauss points for the given element

                    for ig in range(ngauss):
                        #                        print ig+1

                        # ???????????????? Transfer deformation gradient to the previous value ?????????

                        # ---------------------- F_t = F_tau previous -------------

                        F_t = garray_t[ielem.index][ig].defg

                        SHP = SHP3D(gaussw[0][ig], gaussw[1][ig], gaussw[2][ig], XYZ)

                        F_tau = calc_F(xyz, SHP[0])

                        ########### Plastic Defomation Gradient at time = t should be here ######

                        Fp_t = garray_t[ielem.index][ig].pdefg

                        # ------------------- Calculate matrix [A]
                        A_mat = calc_A(F_tau, Fp_t)

                        # ------------------  Calculate matrix [B_alpha]
                        B_alpha = calc_B(n_slip, A_mat, schmid)

                        # ----------------- Calculate the trial elastic stress S_trial
                        S_trial = stress_trial(C_mat, A_mat)

                        # ---------------- Calculate the matrix C_alpha for each slip system
                        C_alpha = calc_C_alpha(n_slip, C_mat, B_alpha)

                        ######################## Enter the User Material ###################

                        res_ssd = garray_t[ielem.index][ig].res_ssd
                        dgam = garray_t[ielem.index][ig].dgam
                        dgam_dta = garray_t[ielem.index][ig].dgam_dta
                        S_star0 = garray_t[ielem.index][ig].S_star

                        U = UMAT(S_trial, C_mat, Fp_t, C_alpha, schmid, F_t, F_tau, res_ssd, dgam, dgam_dta, props,
                                 S_star0)
                        U1 = U.itr()
                        U2 = U.W_mat()

                        strs[ig][0] = U1[2][0][0];
                        strs[ig][1] = U1[2][1][1];
                        strs[ig][2] = U1[2][2][2]
                        strs[ig][3] = U1[2][0][1];
                        strs[ig][4] = U1[2][1][2];
                        strs[ig][5] = U1[2][0][2]

                        #####################     U1[0]=Fe_tau, U1[1]=Fp_tau, U1[2]=Cauchy, U1[3]=res_ssd, U1[5]=dgam , U1[6]=dgam_dta
                        mg = array(ielem.index, ig, U1[2], F_tau, U1[1], U2, U1[3], U1[5], U1[6], U1[7])

                        array_tau.append(mg)
                    ########### End of gauss point loop ##############

                    ############### Element reaction forces or internal forces ################

                    InF = element_re(ngauss, nnode, xyz, strs)

                    lnode = [ielem.nodes[i].inode for i in range(nnode)]

                    gf_nod = asembl_vec(gf_nod, InF, lnode)

                    ############## End of Element reaction forces or internal forces ################

                    garray_tau.append(array_tau)

                    ############# Calculate the stiffness matrix ###############

                    dep = [garray_tau[ielem.index][ig].dep for ig in range(ngauss)]

                    ev = STIF3D(ngauss, nnode, xyz, strs, dep)
                    estiff = ev[0]
                    vol_g.append(ev[1])

                    ######## asembel total stiffness matrix ##############
                    for i in range(nnode):
                        for j in range(nnode):
                            for k in range(3):
                                for l in range(3):
                                    gstiff[3 * ielem.nodes[i].inode + k][3 * ielem.nodes[j].inode + l] += \
                                    estiff[3 * i + k][3 * j + l]
                # ----------------- end of element calculations ----------------------
                for i in range(neq):
                    unb_nod[i] = -gf_nod[i]
                ########## Check for L2 norm of unb_nod for convergence ##############

                for ibc in range(nbc):
                    jbc = 3 * kbc_n[ibc] + kbc_d[ibc] - 1
                    unb_nod[jbc] = 0.0

                norm_gs = math.sqrt(dot_product(gf_nod, neq))
                norm_us = math.sqrt(dot_product(unb_nod, neq))

                ratio_norm = norm_us / norm_gs

                for i in range(neq):
                    unb_nod[i] = -gf_nod[i]

                print
                istep + 1, iconv + 1, norm_us, ratio_norm

                iconv += 1
            #            raw_input()

            strsstrnv = user_print(tnode, nelem, ngauss, gdisp, garray_tau, vol_g)
            strsv = strsstrnv[0];
            strnv = strsstrnv[1]
            strsvol.append(strsv)
            strnvol.append(strnv)

            #            if istep == nstep-1:
            for iel in range(nelem):
                Cauchy_stress_el = []
                Fp_tau_el = []
                Ftau_el = []

                for ig in range(ngauss):
                    Cauchy_stress_el.append(garray_tau[iel][ig].cauchy)
                    Ftau_el.append(garray_tau[iel][ig].defg)
                    Fp_tau_el.append(garray_tau[iel][ig].pdefg)
                Cauchy_stress.append(Cauchy_stress_el)
                Ftau.append(Ftau_el)
                Fp_tau.append(Fp_tau_el)

        return strnvol, strsvol, gdisp, Cauchy_stress, Fp_tau, Ftau


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# ---------------- reading input file --------------------
def main(f1_list):
    f = open('py.15', 'r')
    a = [int(x) for x in f.readline().split()]

    elements = []
    nnode = a[0];
    nelem = a[1];
    tnode = a[2];
    nbc = a[3];
    nconv = a[4] - 1;
    n_slip = a[5]  # nstep = a[4] ;
    for i in range(nelem):
        elements.append([int(x) - 1 for x in f.readline().split()])

    coors = []
    for i in range(tnode):
        b = [float(x) for x in f.readline().split()]
        b[0] = int(b[0]) - 1
        coors.append(b)

    bcs = []
    for i in range(nbc):
        bcs.append([int(x) for x in f.readline().split()])

    # angle = [float(x) for x in f.readline().split()]

    props = [float(x) for x in f.readline().split()]

    tstep = [float(x) for x in f.readline().split()]
    tstep[0] = int(tstep[0])
    nstep = tstep[0]
    props.append(tstep[1])
    # ----------------------------------------------
    # f1 = open('texture.15', 'r')
    # angle = []
    # for i in range(nelem):
    #     angle.append([float(x) for x in f1.readline().split()])

    angle = []
    for i in range(nelem):
        angle.append([float(x) for x in f1_list])

    # ---------------- end of reading input file --------------------

    ssy = open('py.22', 'w')
    # disp = open('py.01','w')

    ft = open('F_tau.txt', 'w')

    fpt = open('Fp_tau.txt', 'w')

    start = time.time()

    m = Mesh(nnode, tnode, coors, nelem, elements)
    ga = m.solve(nbc, bcs, angle, props, tstep, nconv, n_slip)

    F_tau_ave_el = [[0. for ii in range(3)] for jj in range(3)]
    F_tau_ave = []

    Fp_tau_ave_el = [[0. for ii in range(3)] for jj in range(3)]
    Fp_tau_ave = []
    ngauss = 8

    for istep in range(nstep):

        F_tau_ave_el = [[0. for ii in range(3)] for jj in range(3)]
        Fp_tau_ave_el = [[0. for ii in range(3)] for jj in range(3)]

        for ig in range(ngauss):
            for i in range(3):
                for j in range(3):
                    F_tau_ave_el[i][j] += ga[5][istep][ig][i][j]
                    Fp_tau_ave_el[i][j] += ga[4][istep][ig][i][j]

        for i in range(3):
            for j in range(3):
                F_tau_ave_el[i][j] = F_tau_ave_el[i][j] / ngauss
                Fp_tau_ave_el[i][j] = Fp_tau_ave_el[i][j] / ngauss

        F_tau_ave.append(F_tau_ave_el)
        Fp_tau_ave.append(Fp_tau_ave_el)

    print(time.time()-start)

    for istep in range(nstep):
        # print(ft, F_tau_ave[istep][0][0], '  ', F_tau_ave[istep][0][1], '  ', F_tau_ave[istep][0][2], '  ', \
        # F_tau_ave[istep][1][0], '  ', F_tau_ave[istep][1][1], '  ', F_tau_ave[istep][1][2], '  ', F_tau_ave[istep][2][
        #     0], '  ', F_tau_ave[istep][2][1], '  ', F_tau_ave[istep][2][2])
        #
        # print(fpt, Fp_tau_ave[istep][0][0], '  ', Fp_tau_ave[istep][0][1], '  ', Fp_tau_ave[istep][0][2], '  ', \
        # Fp_tau_ave[istep][1][0], '  ', Fp_tau_ave[istep][1][1], '  ', Fp_tau_ave[istep][1][2], '  ', Fp_tau_ave[istep][2][
        #     0], '  ', Fp_tau_ave[istep][2][1], '  ', Fp_tau_ave[istep][2][2])

        print(F_tau_ave[istep][0][0], '  ', F_tau_ave[istep][0][1], '  ', F_tau_ave[istep][0][2], '  ',
              F_tau_ave[istep][1][0], '  ', F_tau_ave[istep][1][1], '  ', F_tau_ave[istep][1][2], '  ',
              F_tau_ave[istep][2][0], '  ', F_tau_ave[istep][2][1], '  ', F_tau_ave[istep][2][2], file=ft)

        print(Fp_tau_ave[istep][0][0], '  ', Fp_tau_ave[istep][0][1], '  ', Fp_tau_ave[istep][0][2], '  ',
              Fp_tau_ave[istep][1][0], '  ', Fp_tau_ave[istep][1][1], '  ', Fp_tau_ave[istep][1][2], '  ',
              Fp_tau_ave[istep][2][0], '  ', Fp_tau_ave[istep][2][1], '  ', Fp_tau_ave[istep][2][2], file=fpt)

    fpt.close()

    # --------------- volume stress-strain--------------
    x = []
    y = []
    for istep in range(nstep):
        x.append(ga[0][istep][1])
        y.append(ga[1][istep][1] / 1.0e6)
        ssy.write("%f   %f" % (ga[0][istep][1], ga[1][istep][1] / 1.0e6))
        ssy.write('\n')

    ssy.close()
    # --------------- end of volume stress-strain--------------
    # ---------------- Cauchy stress sigmaxx ------------
    strxx = [0.0 for i in range(tnode)]
    stryy = [0.0 for i in range(tnode)]
    strzz = [0.0 for i in range(tnode)]

    for inode in range(tnode):
        jcount = 0
        strxx[inode] = 0.0

        for iel in range(nelem):
            lnode = elements[iel]
            for i in range(8):
                if lnode[i + 2] == inode:
                    jcount = jcount + 1
                    strxx[inode] += ga[3][iel][i][0][0]
                    stryy[inode] += ga[3][iel][i][1][1]
                    strzz[inode] += ga[3][iel][i][2][2]

        strxx[inode] = strxx[inode] / jcount
        stryy[inode] = stryy[inode] / jcount
        strzz[inode] = strzz[inode] / jcount

    sx = open('py.11', 'w')

    for inode in range(tnode):
        sx.write("%f  %f  %f  %.8f  %.8f  %.8f  %f  %f  %f" % (coors[inode][1], coors[inode][2], coors[inode][3], \
                                                               ga[2][inode][0], ga[2][inode][1], \
                                                               ga[2][inode][2], strxx[inode], stryy[inode], \
                                                               strzz[inode]))
        sx.write('\n')

    for iel in range(nelem):
        lnode = elements[iel]
        for j in range(8):
            sx.write(str(lnode[j + 2] + 1))
            sx.write('  ')
        sx.write('\n')

    sx.close()

    # ---------------- end of Cauchy stress sigmaxx ------------
    #
    # plot(x, y)
    # ylabel('Stress(MPa)')
    # xlabel('Strain')
    # show()



