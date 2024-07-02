import numpy as np
from scipy.interpolate import splrep, splev, bisplrep, bisplev
import matplotlib.pyplot as plt
import re

a=1
def read_nas_input(filename):
    Ma_list_DLM = []
    k_list = []

    num = r"\d+\.\d*E?[+-]?\d*"  # regular expression for floats in this context

    with open(filename) as input_file:
        for row in input_file:
            if row[0:4] == 'AERO':
                scalars = re.findall(num, row)
                l_ref = float(scalars[1])/2
                rho = float(scalars[2])
            elif row[0:7] == 'MKAERO1':
                Ma_list_DLM += re.findall(num, row[4:])
            elif row[0:3] == '+MK':
                k_list += re.findall(num, row[4:])
            elif row[0:6] =='PAERO1':
                break
    Ma_list_DLM = list(set(Ma_list_DLM))  # removes duplicates
    Ma_list_DLM = list(map(float, Ma_list_DLM))  # convert string to float
    k_list = list(map(float, k_list))  # convert string to float
    return Ma_list_DLM, k_list, l_ref, rho


def read_fort(filename):
    # read a matrix from a nastran file. Line by line and assemble matrix rows

    my_data = []
    info = [] 
    matrow = [] 
    i = 0
    num = r"-?0.\d+[DE][-+]\d{2}"  # regular expression for a double in fortran
    singleint = r"[\s-]\d+"  # regular expression for an integer

    with open(filename) as fort:
        for row in fort:  # read file line-by line
            if row[0:3] != '   ':  # if the row contains numerical data
                row = re.sub('D', 'E', row)  # replace all Ds indicating double format
                row = re.sub('\s', '', row)  # remove whitespaces
                matrow += re.findall(num, row)   
            elif matrow:  # if not numerical and if matrow is not empty
                info += [list(map(int, re.findall(singleint, row)))]
                my_data += [list(map(float, matrow))]
                matrow = []
            else:
                info += [list(map(int, re.findall(singleint, row)))]
    return info, my_data


def rebuild4(info, my_data, nMa, nk):
    info.pop()  # remove last row
    firstline = info.pop(0)  # remove first row
    n = int(firstline[1])  # of modes

    final = np.zeros((nMa, nk, n, n), dtype=complex)  
    for i0 in range(nMa):
        for i1 in range(nk):
            for i2 in range(n):
                temp = my_data[i0 * n * nk + i1 * n + i2]
                temp = list(map(complex, temp[0::2], temp[1::2]))
                final[i0, i1, :, i2] = temp
    return final


def rebuild2(my_data):
    return np.diag(np.squeeze(my_data))


def get_data():  # get input matrix data and build matrices line by line
    pathQHHL = 'Data/fort.14'
    pathKHH = 'Data/fort.15'
    pathMHH = 'Data/fort.16'
    pathBHH = 'Data/fort.17'
    input_file = 'Data/sb14.nas'
    Ma_list_DLM, k_list, l_ref, rho = read_nas_input(input_file)
    infoQ, datQ = read_fort(pathQHHL)
    infoK, datK = read_fort(pathKHH)
    infoM, datM = read_fort(pathMHH)
    infoB, datB = read_fort(pathBHH)
    QHHL = rebuild4(infoQ, datQ, len(Ma_list_DLM), len(k_list))
    KHH = rebuild2(datK)
    MHH = rebuild2(datM)
    BHH = rebuild2(datB)
    return QHHL, KHH, MHH, BHH, Ma_list_DLM, k_list, l_ref, rho


def create_splines(QHHL, Ma_list_DLM, k_list):
    """
    Creates a Spline for each entry of Qij(Ma,k)
    Fully interpolating splines are used. No error is permitted.
    Linear Interpolation, lower degree minimizing splines or a polynomial representation are probably also sufficient
    Dimensions of Q: (n_Ma, n_k, n_modes, n_modes)
    """

    order_itp_Ma = min(3, len(Ma_list_DLM) - 1)  
    order_itp_k = min(3, len(k_list) - 1)  
    Q_spline_real = np.empty(QHHL.shape[2:], dtype=object)  
    Q_spline_imag = np.empty(QHHL.shape[2:], dtype=object)  

    if order_itp_k and order_itp_Ma:  
        for i, j in np.ndindex(QHHL.shape[2:]):
            xij, yij = np.meshgrid(Ma_list_DLM, k_list)  
            Aij = QHHL[:, :, i, j].flatten() 
            xij = xij.flatten()
            yij = yij.flatten()  
            Q_spline_real[i, j] = bisplrep(x=xij, y=yij, z=np.real(Aij), kx=order_itp_Ma, ky=order_itp_k)
            Q_spline_imag[i, j] = bisplrep(x=xij, y=yij, z=np.imag(Aij), kx=order_itp_Ma, ky=order_itp_k)
    elif not order_itp_Ma:  # if only one Mach number was given to nastran for DLM
        for i, j in np.ndindex(QHHL.shape[2:]):
            Aij = QHHL[:, :, i, j].squeeze()  # get all points for Qij in one vector
            Q_spline_real[i, j] = splrep(x=k_list, y=np.real(Aij), k=order_itp_k)
            Q_spline_imag[i, j] = splrep(x=k_list, y=np.imag(Aij), k=order_itp_k)
    elif not order_itp_k:   # if only reduced frequency was given to nastran for DLM
        for i, j in np.ndindex(QHHL.shape[2:]):
            Aij = QHHL[:, :, i, j].squeeze()  # get all points for Qij in one vector
            Q_spline_real[i, j] = splrep(x=Ma_list_DLM, y=np.real(Aij), k=order_itp_k)
            Q_spline_imag[i, j] = splrep(x=Ma_list_DLM, y=np.imag(Aij), k=order_itp_k)
    return Q_spline_real, Q_spline_imag


def interpolate_Aer(Q_spline, Ma, k, nMa, nk):
    Q_spline_real, Q_spline_imag = Q_spline  
    Q = np.zeros(Q_spline_real.shape, dtype=complex)  
    if nMa-1 and nk-1:
        for i, j in np.ndindex(Q.shape):
            Q[i, j] += bisplev(Ma, k, Q_spline_real[i, j])
            Q[i, j] += bisplev(Ma, k, Q_spline_imag[i, j]) * 1j
    elif k-1:
        for i, j in np.ndindex(Q.shape):
            Q[i, j] += splev(k, Q_spline_real[i, j])
            Q[i, j] += splev(k, Q_spline_imag[i, j]) * 1j
    elif Ma-1:
        for i, j in np.ndindex(Q.shape):
            Q[i, j] += splev(Ma, Q_spline_real[i, j])
            Q[i, j] += splev(Ma, Q_spline_imag[i, j]) * 1j
    return Q


def flutter_equation(M, D, K, Q, p, V, rho, l_ref):
    A = (V / l_ref) ** 2 * M
    B = (D - rho / 2 * V * np.imag(Q) * l_ref / p.imag) * (p / l_ref * V) + K - rho / 2 * V ** 2 * np.real(Q)
    return np.linalg.solve(A, B)


def solve_eig(A, lam0, v0, tol, max_it):
    """
    Rayleigh Quotient Iteration (Inverse Power method)
    Used for convergence to nearest to lam0,v0 eigenvalue/vector
    """

    it = 0  
    residual = np.inf
    while residual > tol and it < max_it:  
        v1 = np.linalg.solve((A - lam0 * np.identity(A.shape[0])), v0)  
        v1 = v1 / np.linalg.norm(v1)
        lam1 = (v1.T @ A @ v1) / (v1.T @ v1)

        # The matrix is not invertible, when lam0 is equal to
        # the actual eigenvalue of A, thus lam1 will be nan
        if np.isnan(lam1):  
            return lam0, v0

        residual = abs(lam0 - lam1)  # refresh residual
        it += 1
        v0 = v1  # refresh eigenvector estimate
        lam0 = lam1  # refresh eigenvalue estimate
    return lam1, v1  


def mode_participation(q, Q):  # Jentys p8
    """
    Find modal participation measured in terms of power
    """
    Xi = np.diag(q)
    P = np.imag(Xi.T.conjugate() @ Q @ Xi)
    MP = np.sum(np.abs(P), axis=1)
    MP = MP / max(MP)
    return MP


def find_flutter(p, q, V_list, nMa, nk):
    """
    Find flutter boundary index
    i refers to the mode index
    j refers to the speed index
    """
    flutter_data = []  # initialize flutter data list
    for i in range(p.shape[1]):  # mode loop
        # in j-loop index 1, corresponding to the first speed is ignored, there shouldn't be flutter at the first speed
        # and the code is simpler to write and read.
        for j in range(1, p.shape[0]):  # V-loop
            if p[j, i].real > 0:  # negative damping
                # linear interpolation between first detected flutter speed and previous (at p.real=0)
                alpha = (p[j - 1, i].real - 0) / (p[j, i].real - p[j - 1, i].real)  # factor
                V_flutter = V_list[j - 1] + alpha * (V_list[j - 1] - V_list[j])
                p_flutter = p[j - 1, i] + alpha * (p[j - 1, i] - p[j, i])
                q_flutter = q[j - 1, i] + alpha * (q[j - 1, i] - q[j, i])  
                k_flutter = abs(p_flutter.imag)
                Q = interpolate_Aer(Q_spline, V_list[j], k_flutter, nMa, nk)
                MP = mode_participation(q_flutter, Q)
                flutter_data += [[V_flutter, i, p_flutter, MP, q_flutter]]  # append flutter data
                break  # after flutter speed for a mode is found, there is no need to
                # look for others, so proceed with next mode
    return flutter_data


def plot_d_f(V_list, p, l_ref, omega_s, d_s, debug=False):
    d = -np.real(p) / np.abs(p) * 100  # JS 7.17
    f = np.expand_dims(V_list, 1) * np.imag(p) / 2 / np.pi / l_ref  
    V_list = V_list * 3.6  # switch to m/s from km/h
    if debug:  # add structural solutions for comparison
        d_s = d_s / 2 / omega_s * 100  # JS 4.39
        f0 = np.array(np.sqrt(omega_s) / 2 / np.pi)
        d = np.vstack((d_s.T, d))  # insert structural solutions
        f = np.vstack((f0.T, f))  # insert structural solutions
        V_list = np.insert(V_list, 0, 0)  # insert zero at the beginning of the speed vector
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, num='Results')  # stack plots vertically
    for i in range(p.shape[1]):  # a line for each mode
        ax1.plot(V_list, d[:, i], linewidth=.7)
        ax2.plot(V_list, f[:, i], linewidth=.7)
    ax1.set_ylabel('Damping')  
    ax1.set_ylim(-2, 12)
    ax1.axhline(color='k', linestyle=':')  # add a dotted line at y=0 to indicate the flutter boundary
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Speed in km/h')
    ax1.minorticks_on()
    ax1.grid(True, 'both')
    ax1.grid(True, 'minor', c='0.9')
    ax2.minorticks_on()
    ax2.grid(True, 'both')
    ax2.grid(True, 'minor', c='0.9')
    plt.savefig("results_" + str(len(V_list)) + ".pgf")
    plt.show()  


## Preprocessing
debug = False  # more detailed logging
QHHL, K, M, D, Ma_list_DLM, k_list, l_ref, rho = get_data()  # get Data from NASTRAN DLM

# additional data
tol = 1e-6  # tolerance for eigenvalue and pk convergence (could be obtained from FLUTTER EPS)
max_it = 10  # maximum iterations for pk-loop and eigenvalue loop
c = 343  # speed of sound
V0 = 10  # first speed in m/s (could be obtained from FLFACT)
Vl = 370 / 3.6  # last speed in m/s (could be obtained from FLFACT)
nV = 100  # number of speed increments (could be obtained from FLUTTER NVALUE)
n_modes = M.shape[0]
V_list = np.linspace(V0, Vl, nV)  # list of speeds to be explored
p_results = np.zeros((len(V_list), n_modes), dtype=complex)  # eigenvalue results
q_results = np.zeros((len(V_list), n_modes, n_modes), dtype=complex)  # eigenvector results


## pk-method
omega, q_start = np.linalg.eig(K)  # static case eigenvalues and eigenvectors
p_start = (np.diag(D) / omega / 2 + 1j * np.sqrt(omega)) * l_ref / V_list[0]  # starting value for p
p_results[-1, :] = p_start  # start values saved in last row of array
q_results[-1, :] = q_start  # start values saved in last row of array

# create (bivariant) splines (not bivariant if only one Mach number is considered)
Q_spline = create_splines(QHHL, Ma_list_DLM, k_list)

# pk-loop
for indexV, V in enumerate(V_list):  
    for index_mode in range(n_modes):   

        p0 = p_results[indexV - 1, index_mode]  
        q0 = q_results[indexV - 1, index_mode]
        residual = np.inf  
        it = 0

        while residual > tol and it < max_it:  
            k = abs(p0.imag)  # failsafe
            Q = interpolate_Aer(Q_spline, V/c, k, len(Ma_list_DLM), len(k_list))  
            F = flutter_equation(M, D, K, Q, p0, V, rho, l_ref)
            lam0 = -p0 ** 2  # JS 7.19 -- minus switches complex and real part later
            lam1, q = solve_eig(F, lam0, q0, tol, max_it)
            residual = abs(lam1 - lam0)  # does it converge?
            it += 1
            p0 = np.sqrt(lam1) * 1j  # determine root and switch back complex and real part
            q0 = q

        p_results[indexV, index_mode] = p0  
        q_results[indexV, index_mode] = q

        if debug:
            print(indexV, index_mode, 'p0:', p0)  # in debug mode, the resulting eigenvalues are shown
        else:
            progress = round((indexV * n_modes + index_mode+1) / len(V_list) / n_modes * 100, 1)
            print('\r' + str(progress) + '%', end='')  # out of debug mode, the progress displayed in %

## Postprcessing
# find flutter speeds, modes and modal participation
flutter_data = find_flutter(p_results, q_results, V_list, len(Ma_list_DLM), len(k_list))

# create d and f plots
plot_d_f(V_list, p_results, l_ref, omega, np.diag(D), debug)

# Note: JS page 9: Flutter modes with k>1 do not matter
