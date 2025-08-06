function [Hobj, psi_target] = parity_separated(L, st, bPrint, sector)

if nargin < 3
    bPrint = true;
end

if nargin < 4
    sector = 'even';
end

% bPrint = true;
% st = 1;
% sector  = 'even';      % 'odd' (periodic fermions) .... 'even' (anti-periodic)
wantOdd = strcmp(sector,'odd');

% momentum  k = 2π(n+shift)/L ,  shift = 0 (odd) or 1/2 (even)
shift = double(~wantOdd)*0.5;
qvals = 2*pi/L * ( (0:L-1) + shift );          
F = exp(1i*(0:L-1).' * qvals) / sqrt(L);    


[c,cd] = create_fermionic_ops(L);

% momentum-space fermions  c_q , c^dagger_q   and  number operators  n_q
c_q  = cell(1,L);   cd_q = cell(1,L);   Nqs  = cell(1,L);
for q = 1:L
    op_cq  = sparse(0);   op_cdq = sparse(0);
    for j = 1:L
        op_cq  = op_cq  + conj(F(j,q)) * c{j};
        op_cdq = op_cdq + F(j,q)  * cd{j};
    end
    c_q{q}  = op_cq;
    cd_q{q} = op_cdq;
    Nqs{q}  = op_cdq * op_cq;
end

% Hamiltonian
Hkin = sparse(2^L,2^L);
for q = 1:L
    Hkin = Hkin + cos(qvals(q)) * Nqs{q};
end
% auxiliary Hamiltonian
% Haux = Hkin + 10^{-1} n_{2nd} + 10^{-2} n_{3rd} + ...

Haux = Hkin;
w  = 1e-1; 
for q = 1:L           
    Haux = Haux + w * Nqs{q};
    w    = w * 1e-1;   
end


[V,~]   = eig(full(Haux));
extract = @(A) diag(V' * A * V);

Nqvals  = cell2mat( cellfun(extract, Nqs, 'uni',0) );
Nqvals   = round(Nqvals);
Ntotvals = sum(Nqvals,2);


keep = (mod(Ntotvals,2)==wantOdd);
rows = find(keep);
Nq_phys = Nqvals(rows,:);
E_phys = Nq_phys * cos(qvals).';     % E = \sum_k cos k n_k
[E_sorted,ix] = sort(E_phys);
rows_sorted  = rows(ix);



if st == 0
    psiTargets = V(:, rows_sorted);              % 2^L × 2^(L-1)

    % ---- build all Hobj -------------------------------------------------
    Id    = speye(2^L);
    nStates = size(psiTargets,2);                % = 2^(L-1)
    Hobjs   = cell(1,nStates);
    for k = 1:nStates
        targetRow = rows_sorted(k);
        h = sparse(2^L,2^L);
        for q = 1:L
            h = h + (Nqs{q} - Nqvals(targetRow,q)*Id)^2;
        end
        Hobjs{k} = h;
    end

    % ---- save the file -----------------------------------------------------------
    fname = sprintf('cache_L%d_%s.mat', L, sector);
    save(fname,'psiTargets','Hobjs','qvals','E_sorted','-v7.3');
    if bPrint, fprintf(' cache written to %s  (%d states)\n',fname,nStates); end

    Hobj = [];  psi_target = [];
    return
end

psi_target = V(:, rows_sorted(st));

if bPrint
    hdr = 'Rows: (Row# ';
    for k = 0:L-1
        label = qvals(k+1)/pi;            % in units of pi
        if abs(label - round(label)) < 1e-12
            txt = sprintf('n%dπ ', round(label));
        else
            [num,den] = rat(label);
            txt = sprintf('n%dπ/%d ', num, den);
        end
        hdr = [hdr txt];
    end
    hdr = [hdr '| Ntot |    E)'];
    fprintf('%s\n', hdr);

    T = real([ rows_sorted, ...
          Nqvals(rows_sorted,:), ...
          Ntotvals(rows_sorted), ...
          real(E_sorted) ]);
    disp(T);
end

%cost Hamiltonian  Hobj  = Σ_q (n_q - n_q^target)^2
Id   = speye(2^L);
targetRow = rows_sorted(st);
Hobj = sparse(2^L,2^L);
for q = 1:L
    Hobj = Hobj + ( Nqs{q} - Nqvals(targetRow,q)*Id )^2;
end

%% ------------------------------------------------------------------------
function [c,cd] = create_fermionic_ops(L)
    sz     = sparse([1 2],[1 2],[1 -1],2,2);
    sminus = sparse(2,1,1,2,2);
    I2     = speye(2);
    c  = cell(1,L);   cd = cell(1,L);
    for j = 1:L
        op = 1;
        for site = 1:L
            if   site < j, op = kron(op,-sz);
            elseif site==j, op = kron(op,sminus);
            else            
                op = kron(op,I2);
            end
        end
        c{j}  = sparse(op);
        cd{j} = c{j}';
    end
end

end