L = 6;
sector  = 'even';
p = 7;  % QAOA depth

nTrials = 1000;

% loading file specified by L and sector
cachefile = sprintf('cache_L%d_%s.mat', L, sector);
load(cachefile, 'psiTargets', 'Hobjs');

% for each sector, the number of states are 2^L / 2
statesToTry = 1:(2^L/2);


%% n_angles to try

%  spinning angles -- works for L=4 with p_max = 4
%zargD = exp(1i*pi/2*(0:L-1));

% for L=6 odd sector -- did not work at p=7/ntrials=1000 for st = 4,5,8,11,13,15,21,23,24,28,29
% for L=6 even sector - did not work at p=7/ntrials=1000 for st = 5,6,7,8,11,12,21,22,25,26,27,28
zargD = exp(1i*2*pi*(0:L-1)/L);

% zargD = exp(1i*pi/2*ones(L,1)); % uniform angles
%zargD = exp(1i*2*pi/L*randi(L,L,1)); % random quantized angles - bad for L=6

n_angles          = zeros(L,2);
n_angles(:,1)     = pi/2;
n_angles(:,2)     = angle(zargD);


%% Set up Hamiltonians and evolution operators


options = optimoptions('fminunc', 'GradObj', 'on', 'Hessian', 'off', ...
    'Display', 'off', 'TolX', 1e-6, 'TolFun', 1e-8, ...
    'Algorithm', 'quasi-newton', ...
    'MaxFunEvals', Inf, 'MaxIter', Inf);

for st = statesToTry
    fprintf('===== Eigenstate %d =====\n', st)
    psi_target = psiTargets(:, st);
    Hobj       = Hobjs{st};

    
    [QAOAhelperfcn, HamObj, HamA, HamB, HamC, HamD, EvolA, EvolB, EvolC, EvolD] = ...
        Setup4ExcitedStateQAOA(L, Hobj, n_angles);
    
    % initialization

    myfun = @(param) QAOAhelperfcn(p, param);

    bestfval = inf;
    param_opts = cell(nTrials, 1);
    fvals = nan(nTrials, 1);
    tic;
    for ind = 1:nTrials
        param0 = (2 * rand(p,4) - 1) * pi/2;
        [param_opts{ind}, fvals(ind)] = fminunc(myfun, param0, options);
        if fvals(ind) < bestfval 
            bestfval = fvals(ind);
        end
        % fprintf('Trial %d/%d, current energy = %0.6e, best energy = %0.6e\n', ind, nTrials, fvals(ind), bestfval);
    end
    [bestfval, Imin] = min(fvals);
    param_opt = param_opts{Imin};
    
    % Evaluate final state
    param_mat_opt = reshape(param_opt,[],4);   % restore matrix shape
    [F_opt, ~, psi_final] = QAOAhelperfcn(p, param_mat_opt);
    
    % Compare with exact eigenstate
    fidelity = abs(psi_final' * psi_target)^2;
    fprintf('-- Finished after %0.2f s\n', toc);
    
    fprintf('Final cost: %.8f\n', bestfval);
    fprintf('# repetitions of global minimum = %d/%d \n', nnz(fvals<=bestfval + 1e-12), nTrials);
    fprintf('Infidelity with target state: %.6e\n', 1-fidelity);
end