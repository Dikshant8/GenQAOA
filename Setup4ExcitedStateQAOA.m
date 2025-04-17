function [QAOAhelperfcn, HamObj, HamA, HamB, HamC, HamD, ...
     EvolA, EvolB, EvolC, EvolD] =  Setup4ExcitedStateQAOA(L, Hobj, n_angles, m_angles)

% Inputs:
%   L         - number of lattice sites
%   O_k       - operator O_k in computational basis
%   lambda_k  - target eigenvalue of O_k to isolate
%   n_angles  - Nx2 array of (theta, phi) defining HamD directions
%   m_angles  - optional Nx2 array for initial product state (default = n_angles)
%
% Outputs:
%   QAOAhelperfcn - evaluation + gradient function for QAOA
%   HamObj        - (O_k - lambda_k I)^2 objective Hamiltonian
%   HamA-D        - driver Hamiltonians
%   EvolA-D       - corresponding evolution functions

if nargin < 7
    m_angles = n_angles;
end

%% Pauli matrices
sx = sparse([0,1; 1,0]);
sy = sparse([0,-1i; 1i, 0]);
sz = sparse(diag([1,-1]));

%% Construct HamObj
HamObj = Hobj;

%% Driver Hamiltonians

% ZZ driver
HamA = sparse(0);
for j = 1:L
    j2 = mod(j, L) + 1;
    HamA = HamA + Ham2LTerm(sz, sz, j, j2, L);
end
HamAdiag = full(diag(HamA));
EvolA = @(psi, alpha) exp(-1i * alpha * HamAdiag) .* psi;

% HamB
HamB = krondist(sx, L);
EvolB = @(psi, beta) EvolHamB(L, beta, psi, false);

% HamC
HamC = krondist(sz, L);
HamCdiag = full(diag(HamC));
EvolC = @(psi, gamma) exp(-1i * gamma * HamCdiag) .* psi;

% HamD = Pauli axis driver
n_thetas = n_angles(:, 1);
n_phis = n_angles(:, 2);
n_vecs = [sin(n_thetas).*cos(n_phis), sin(n_thetas).*sin(n_phis), cos(n_thetas)];
HamD = krondist(sx, L, n_vecs(:,1)) + ...
        krondist(sy, L, n_vecs(:,2)) + ...
        krondist(sz, L, n_vecs(:,3));
EvolD = @(psi, delta) PauliRotations(L, delta, n_vecs, psi);


% initial state
psi0 = getProductState(m_angles);

QAOAhelperfcn = @(p, param) MultiQAOAGrad(p, HamObj, ...
    {HamA, HamB, HamC, HamD}, param, psi0, ...
    {EvolA, EvolB, EvolC, EvolD});