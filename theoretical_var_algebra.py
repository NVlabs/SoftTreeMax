import numpy as np
from sympy import Matrix
S = 4
A = 3

case_str = [ 'Permutation', 'Random', 'Uniform']
for i_case in range(3):
    def draw_MDP():
        theta = np.random.rand(S)
        #theta[0] += 10
        Q = np.random.rand(S, A)
        P = np.random.rand(S, A, S)
        for s in range(S):
            a_idx = np.random.randint(A)
            s_idx = np.random.randint(S)
        # P[:, :, 0] += 50
        if i_case == 0:
            P = np.exp(10 * P)
        if i_case == 2:
            P = 0.7 * np.ones((S, A, S)) + 0.3 * np.random.rand(S, A, S)
        P = P / np.sum(P, axis=2, keepdims=True)
        R = np.random.rand(S,A) #np.matmul(np.random.rand(S,1), np.ones((1,A))) #np.random.rand(S,A) # generally a function of (s,a), but we assume here pi_u to be uniform
        # Rs = np.ones(A) #np.ones(A)
        # theta = np.zeros(S)
        gamma = .9
        return theta, Q, P, R, gamma


    theta, Q, P, R, gamma = draw_MDP()


    def get_stationary_dist(transition_matrix):
        '''
        Since the sum of each row is 1, our matrix is row stochastic.
        We'll transpose the matrix to calculate eigenvectors of the stochastic rows.
        '''
        transition_matrix_transp = transition_matrix.T
        eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp)
        '''
        Find the indexes of the eigenvalues that are close to one.
        Use them to select the target eigen vectors. Flatten the result.
        '''
        close_to_1_idx = np.isclose(eigenvals, 1)
        target_eigenvect = eigenvects[:, close_to_1_idx]
        target_eigenvect = target_eigenvect[:, 0]
        # Turn the eigenvector elements into probabilites
        stationary_distrib = target_eigenvect / sum(target_eigenvect)
        return stationary_distrib.real


    def compute_eig(d):
        exp_theta = np.exp(theta)
        Ppiu = P.mean(axis=1)
        dpiu = get_stationary_dist(Ppiu)
        if d == 1:
            evr = np.real(np.linalg.eigvals(Ppiu))
            eva = np.abs(np.linalg.eigvals(Ppiu))
            print('reals: {} \n abs: {}'.format((1 / evr), (1 / eva)))
        Rpiu = R.mean(axis=1)
        Rs = R[0, :]
        D_Rs = np.diag(np.exp(Rs))
        D_Rpiu = np.diag(np.exp(Rpiu))
        DP = np.matmul(D_Rpiu, Ppiu)
        Ppower = np.eye(S)
        # for h in range(1, d):
        #     Ppower = np.linalg.multi_dot([Ppower, np.diag(np.exp(gamma ** h * Rpiu)), Ppiu])

        for h in range(1, d):
            Pii = np.random.rand(S, S)
            Pii = Pii / np.sum(Pii, axis=1, keepdims=True)
            Ppower = np.linalg.multi_dot([Ppower, Pii])

        e2, v2 = np.linalg.eig(Ppower)
        return e2


    def compute_trace(d):
        exp_theta = np.exp(theta)
        pib = np.random.uniform(size=A)
        # pib = np.exp(100 * pib)
        if i_case == 0:
            pib[0] += 1000
        pib = pib / sum(pib)
        pib = np.expand_dims(pib, axis=0)
        Ppib = np.sum(np.multiply(P, np.dstack([pib] * S)), axis=1)
        # Ppiu = P.mean(axis=1)
        # dpiu = get_stationary_dist(Ppiu)
        dpib = get_stationary_dist(Ppib)
        if d == 1:
            evr = np.real(np.linalg.eigvals(Ppib))
            eva = np.abs(np.linalg.eigvals(Ppib))
            print('reals: {} \n abs: {}'.format((1 / evr), (1 / eva)))
        # Rpiu = R.mean(axis=1)
        Rpib = np.sum(R * pib.repeat(S, axis=0), axis=1)
        Rs = R[0, :]
        D_Rs = np.diag(np.exp(Rs))
        D_Rpib = np.diag(np.exp(Rpib))
        DP = np.matmul(D_Rpib, Ppib)
        Ppower = np.eye(S)
        for h in range(1, d):
            Ppower = np.linalg.multi_dot([Ppower, np.diag(np.exp(gamma ** h * Rpib)), Ppib])
        # Ppower = np.linalg.matrix_power(DP, d-1)
        Ps = P[0]

        Psd = np.linalg.multi_dot([D_Rs, Ps, Ppower])

        pid = np.zeros((S, A))
        for s in range(S):
            Ps_var = P[s]
            Psd_var = np.matmul(Ps_var, Ppower)
            pid[s, :] = np.matmul(Psd_var, exp_theta) / np.linalg.multi_dot([np.ones(A), Psd_var, exp_theta])
        Ppid = np.sum(np.multiply(P, np.dstack([pid] * S)), axis=1)
        dpid = get_stationary_dist(Ppid)
        pid_s = pid[0]
        Dpi = np.diag(pid_s)



        Pe_diag = np.diag(np.matmul(Psd, exp_theta))
        np.linalg.eigvals(np.linalg.multi_dot([Psd, Psd.transpose(), np.eye(A) - np.ones((A, A)) / A]))
        inv_diag = np.linalg.matrix_power(Pe_diag, -1)
        # prod = np.linalg.multi_dot([np.diag(exp_theta), np.transpose(Pbar), inv_diag, Pbar, np.diag(exp_theta)])
        # prod1 = np.linalg.multi_dot([inv_diag, Psd, np.diag(exp_theta)]) #Msd
        # prod2 = np.linalg.multi_dot([np.ones((A, A)), Pbar, np.diag(exp_theta)]) / np.linalg.multi_dot([np.ones(A), Pbar, exp_theta])
        # grad = prod1 - prod2
        scaling_factor = np.linalg.multi_dot([np.ones(A), Psd, exp_theta])
        diag_diff = (np.diag(1 / pid_s) - np.ones((A, A))) #/ scaling_factor
        # diag_diff = (np.eye(A) - np.transpose(np.asarray([pid_s] * A)))  # / scaling_factor

        suffix_grad = np.matmul(Psd, np.diag(exp_theta))
        # prefix_grad = diag_diff
        prefix_grad = inv_diag - np.ones((A, A)) / scaling_factor #Vsd
        M = np.linalg.multi_dot([prefix_grad, Ps, np.ones((S, 1)), dpib.reshape((1, S)), np.linalg.matrix_power(Ppib, d-1), np.diag(exp_theta)])
        ns = Matrix(M).nullspace()
        grad = np.matmul(prefix_grad, suffix_grad)
        suffix_cov = np.matmul(np.transpose(suffix_grad), suffix_grad)
        prefix_cov = np.matmul(np.transpose(prefix_grad), prefix_grad)
        cov = np.matmul(np.transpose(grad), grad)
        simplified_grad = np.matmul(np.eye(A) - np.ones((A, A)) / A, Psd)
        simplified_cov = np.matmul(np.transpose(simplified_grad), simplified_grad)
        return np.trace(cov), np.trace(prefix_cov), np.trace(suffix_cov), np.trace(simplified_cov), Ppib
        # # mat = np.matmul(diag_diff, suffix_grad)
        # # out = np.linalg.multi_dot([np.transpose(mat), (Dpi - Dpi ** 2), mat]) # (Dpi - Dpi ** 2)
        # mat1 = np.linalg.multi_dot([diag_diff, Dpi, Psd])
        # out1 = np.matmul(np.transpose(mat1), mat1)
        # mat2 = np.linalg.multi_dot([diag_diff, Psd])
        # out2 = np.matmul(np.transpose(mat2), mat2)
        # return np.trace(out1), np.trace(out1) / np.trace(out2), 0, 0


    D = 12
    d_range = np.arange(1, D)

    # eig_Ppower = []
    # for d in d_range:
    #     e = compute_eig(d)
    #     eig_Ppower.append(np.abs(e))

    res_cov = []
    res_pr_cov = []
    res_sf_cov = []
    res_simp_cov = []

    for d in d_range:
        cov, pr_cov, sf_cov, simp_cov, Ppib = compute_trace(d)
        res_cov.append(cov)
        res_pr_cov.append(pr_cov)
        res_sf_cov.append(sf_cov)
        res_simp_cov.append(simp_cov)

    ratio_cov = np.asarray(res_cov[:-1]) / np.asarray(res_cov[1:])
    ratio_pr_cov = np.asarray(res_pr_cov[:-1]) / np.asarray(res_pr_cov[1:])
    ratio_sf_cov = np.asarray(res_sf_cov[:-1]) / np.asarray(res_sf_cov[1:])
    ratio_simp_cov = np.asarray(res_simp_cov[:-1]) / np.asarray(res_simp_cov[1:])

    e, v = np.linalg.eig(Ppib)
    eig_decay = [abs(e[1])**(2*d) for d in range(1, D)]
    eig_decay = np.asarray(eig_decay) * (res_cov[0] / eig_decay[0])
    print(res_cov)
    import matplotlib.pyplot as plt
    plt.plot(d_range, res_cov, label='{}: True variance'.format(case_str[i_case]))
    plt.plot(d_range, eig_decay, label='{}: Variance bound'.format(case_str[i_case]))
    plt.yscale('log')
plt.xlabel('Depth d')
plt.ylabel('SoftTreeMax \nGradient variance ')
plt.legend()
plt.show()
