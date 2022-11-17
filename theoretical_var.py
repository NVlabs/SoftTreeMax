import numpy as np

def compute_var_ratio(S, A, n_samples):
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

    def draw_MDP():
        theta = np.random.rand(S, A)
        # for s in range(S):
        #     theta[s, np.random.randint(A)] += 3
        Q = np.random.rand(S, A)
        P = np.random.rand(S, A, S)
        a_idx = np.random.randint(A)
        s_idx = np.random.randint(S)
        # P[s_idx, a_idx, s_idx] += 50
        P = P / np.sum(P, axis=2, keepdims=True)
        return theta, Q, P

    def get_pi0():
        pi0 = np.exp(theta) / np.sum(np.exp(theta), axis=1, keepdims=True)
        Ppi0 = np.sum(np.multiply(P, np.dstack([pi0] * S)), axis=1)
        rho_pi0 = get_stationary_dist(Ppi0)
        return pi0, Ppi0, rho_pi0

    def get_pi1():
        theta_cap = np.exp(theta).sum(axis=1)
        mat = [np.ones((S, A)) * theta_cap[0]]
        for v in theta_cap[1:]:
            mat.append(np.ones((S, A)) * v)
        theta_cap_mat = np.dstack(mat)
        pi1_numerator = np.sum(np.multiply(P, theta_cap_mat), axis=2)
        P_sum_over_a = np.repeat(a=P.sum(axis=1, keepdims=True), repeats=A, axis=1)
        pi1_denominator = np.sum(np.multiply(P_sum_over_a, theta_cap_mat), axis=2)
        pi1 = np.divide(pi1_numerator, pi1_denominator)
        Ppi1 = np.sum(np.multiply(P, np.dstack([pi1] * S)), axis=1)
        rho_pi1 = get_stationary_dist(Ppi1)
        return pi1, Ppi1, rho_pi1


    theta, Q, P = draw_MDP()
    # Q = np.ones_like(Q) * 0.5
    pi0, Ppi0, rho_pi0 = get_pi0()
    pi1, Ppi1, rho_pi1 = get_pi1()
    # while isinstance(rho_pi0[0], complex):
    #     theta, Q, P = draw_MDP()
    #     pi0, Ppi0, rho_pi0 = get_pi0()
    #     pi1, Ppi1, rho_pi1 = get_pi1()

    # print('stationary distribution: {}'.format(rho_pi0))

    def compute_grad_pi0(s, a):
        e_vec = np.zeros(A)
        e_vec[a] = 1
        return (e_vec - pi0[s]) * Q[s, a]

    def compute_grad_pi1(s, a):
        theta_cs = theta.flatten()
        theta_cs_exp = np.exp(theta_cs)
        theta_cap = np.exp(theta).sum(axis=1)
        denom = np.inner(P[s, a, :], theta_cap)
        prod = np.repeat(P[s,a], A) - pi1[s,a] * np.repeat(P[s].sum(axis=0), A)
        return theta_cs_exp / denom * prod

    def compute_grad(s, a, use_pi0=True):
        return compute_grad_pi0(s, a) if use_pi0 else compute_grad_pi1(s, a)


    def sample_var(use_pi0=True):
        s1, a1 = sample_sa(use_pi0)
        s2, a2 = sample_sa(use_pi0)
        s3, a3 = sample_sa(use_pi0)

        e1 = compute_grad(s1, a1, use_pi0)
        e2 = compute_grad(s2, a2, use_pi0)
        e3 = compute_grad(s3, a3, use_pi0)

        return np.inner(e1, e1) - np.inner(e2, e3)


    def sample_sa(use_pi0=True):
        rho_pi = rho_pi0 if use_pi0 else rho_pi1
        pi = pi0 if use_pi0 else pi1
        s = np.random.multinomial(1, rho_pi)
        s = np.where(s > 0)[0][0]
        a = np.random.multinomial(1, pi[s])
        a = np.where(a > 0)[0][0]
        return s, a


    def estiamte_var():
        v_pi0 = 0
        v_pi1 = 0
        for i in range(n_samples):
            v_pi0 += sample_var(use_pi0=True)
            v_pi1 += sample_var(use_pi0=False)
        return v_pi0 / n_samples, v_pi1 / n_samples

    pi0_var, pi1_var = estiamte_var()
    pi0_bound = np.inner(rho_pi0, (pi0 ** 2).sum(axis=1))
    pi1_bound = np.inner(rho_pi1, (pi1 ** 2).sum(axis=1))
    return pi0_var / pi1_var, pi0_bound / pi1_bound


n_samples = 100
A_res = []
S_res = []
SA_res = []
S_fixed = 10
A_max = 30
A_vec = [i for i in range(3, A_max)]
for A in A_vec:
    A_res.append(compute_var_ratio(S_fixed, A, n_samples)[0])
    print('A={}/{}'.format(A, A_max))
A_fixed = 10
S_max = 30
S_vec = [i for i in range(3, S_max)]
for S in S_vec:
    S_res.append(compute_var_ratio(S, A_fixed, n_samples)[0])
    print('S={}/{}'.format(S, S_max))

for S, A in zip(S_vec, A_vec):
    SA_res.append(compute_var_ratio(S, A, n_samples)[0])
    print('S={}/{}'.format(S, S_max))

import matplotlib.pyplot as plt
plt.plot(S_vec, S_res, label='Pi0 to Pi1 variance ratio')
init_v = 1 #S_res[0] / (S_vec[0] * A_fixed)
plt.plot(S_vec, np.asarray(S_vec) * A_fixed * init_v, label='normalized S * A')
plt.legend()
plt.xlabel('S with A fixed to {}'.format(A_fixed))
plt.figure()
plt.plot(A_vec, A_res, label='Pi0 to Pi1 variance ratio')
init_v = 1 # A_res[0] / (A_vec[0] * S_fixed)
plt.plot(S_vec, np.asarray(A_vec) * S_fixed * init_v, label='normalized S * A')
plt.legend()
plt.xlabel('A with S fixed to {}'.format(S_fixed))
plt.ylabel('Pi0 to Pi1 variance ratio')

plt.show()