%=======================================================================
%  Setup4XXStatesQAOA.m   (revised for momentum-space Hobj)
%=======================================================================
%  Inputs
%     L        – chain length
%     Hobj     – cost Hamiltonian (built outside, e.g. from n_q projectors)
%     s        – length-L vector  s(i)=±1 (bias pattern for HamD)
%     n_angles – L×2  (θ,φ)  axes for optional initial product state
%     m_angles – optional L×2  (θ,φ) ; default = n_angles
%
%  Outputs
%     QAOAhelperfcn  –  handle used by optimiser
%     HamObj         –  the same Hobj (returned for convenience)
%     HamA,B,C,D     –  driver Hamiltonians
%     EvolA,B,C,D    –  exact evolution functions
%=======================================================================

function [QAOAhelperfcn, HamObj, HamA, HamB, HamC, HamD, ...
          EvolA, EvolB, EvolC, EvolD] = ...
          Setup4XXStatesQAOA(L, Hobj, s, n_angles, m_angles)

if nargin < 5                    % default product state angles
    m_angles = n_angles;
end

%%  Pauli matrices
sx = sparse([0 1; 1 0]);
sy = sparse([0 -1i; 1i 0]);
sz = sparse(diag([ 1 -1]));
I2 = speye(2);

%%  === cost Hamiltonian (already built) =============================
HamObj = Hobj;

%%  === Driver layers ===============================================

% 1)  ZZ driver  (periodic)
HamA = sparse(0);
for j = 1:L
    j2 = mod(j, L) + 1;
    HamA = HamA + Ham2LTerm(sz, sz, j, j2, L);   % helper in Leo’s repo
end
HamAdiag = full(diag(HamA));
EvolA    = @(psi, alpha) exp(-1i*alpha*HamAdiag) .* psi;

% 2)  Uniform X mixer
HamB  = krondist(sx, L);                       % Σ_i X_i
EvolB = @(psi, beta) EvolHamB(L, beta, psi, false);

% 3)  Uniform Z phase layer
HamC     = krondist(sz, L);                    % Σ_i Z_i
HamCdiag = full(diag(HamC));
EvolC    = @(psi, gamma) exp(-1i*gamma*HamCdiag).*psi;

% HamD = Pauli axis driver
n_thetas = n_angles(:, 1);
n_phis = n_angles(:, 2);
n_vecs = [sin(n_thetas).*cos(n_phis), sin(n_thetas).*sin(n_phis), cos(n_thetas)];
HamD = krondist(sx, L, n_vecs(:,1)) + ...
        krondist(sy, L, n_vecs(:,2)) + ...
        krondist(sz, L, n_vecs(:,3));
EvolD = @(psi, delta) PauliRotations(L, delta, n_vecs, psi);


%%  === initial product state ========================================
psi0 = getProductState(m_angles);              % helper in repo

%%  === pack helper for optimiser ====================================
QAOAhelperfcn = @(p,param) MultiQAOAGrad( ...
        p, HamObj, {HamA,HamB,HamC,HamD}, param, psi0, ...
                 {EvolA,EvolB,EvolC,EvolD} );
end