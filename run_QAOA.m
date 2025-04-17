% run_QAOA.m
% Assumes initialize.m and Hxx_currents.m already run

p = 8;  % QAOA depth

% Set up Hamiltonians and evolution operators
[QAOAhelperfcn, HamObj, HamA, HamB, HamC, HamD, EvolA, EvolB, EvolC, EvolD] = ...
    Setup4ExcitedStateQAOA(L, Hobj, n_angles);

% initialization
param0 = (2 * rand(p,4) - 1) * pi/2;

% Use fminunc optimizer with gradient
myfun = @(param) QAOAhelperfcn(p, param);

options = optimoptions('fminunc', 'GradObj', 'on', 'Hessian', 'off', ...
    'Display', 'iter', 'TolX', 1e-6, 'TolFun', 1e-8, ...
    'Algorithm', 'quasi-newton', ...
    'MaxFunEvals', Inf, 'MaxIter', Inf, ...
    'PlotFcn', {@optimplotfval, @optimplotstepsize});

[param_opt, fval] = fminunc(myfun, param0, options);

% Evaluate final state
[F_opt, ~, psi_final] = QAOAhelperfcn(p, param_opt);

% Compare with exact eigenstate
fidelity = psi_final' * fid_obj * psi_final;

fprintf('Final cost: %.8f\n', fval);
fprintf('Fidelity with target state: %.6f\n', fidelity);