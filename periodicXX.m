%  periodicXX.m

%  * Builds the complete current algebra   {P_k , Q_k , N_tot}
%    for a periodic XX chain of ANY even length   L = 2,4,6,… .
%  * Finds the simultaneous-eigenbasis.
%  * Prints the table of joint eigen-values.
%  * Constructs cost Hamiltonian  Hobj
%%

clear;  clc;

L  = 4;          %  chain length (must be even)
st = 1;         %  row (1 … 2^L) of the state you want QAOA to prepare

function [c,cd] = create_fermionic_ops(L)
    sz = sparse([1 2],[1 2],[1 -1],2,2);   % σ_z
    sminus = sparse(2,1,1,2,2);            % σ^-
    I2 = speye(2);

    c  = cell(1,L);   cd = cell(1,L);
    for j = 1:L
        op = 1;
        for site = 1:L
            if   site < j, op = kron(op,sz);
            elseif site==j, op = kron(op,sminus);
            else            
                op = kron(op,I2);
            end
        end
        c{j}  = sparse(op);
        cd{j} = c{j}';
    end
end

[c,cd] = create_fermionic_ops(L);


function Jk = build_Jk(cd,c,L,k)      % periodic current
    Jk = sparse(2^L,2^L);
    for j = 1:L
        tgt = mod(j+k-1,L)+1;
        Jk = Jk + cd{j}*c{tgt};
    end
end

%%
% 3)  Construct commuting charges
%       P_k  =  J_k + J_{L-k}
%       Q_k  = (J_k − J_{L-k}) / i
%       P_0  = 2 N_tot

nPairs = L/2 - 1;             
Ps     = cell(1,L/2);         % P_1 … P_{L/2}
Qs     = cell(1,nPairs);      % Q_1 … Q_{L/2-1}


J0   = build_Jk(cd, c, L, 0);      %  ← add cd  c  L
P0   = J0 + J0';
Ntot = 0.5*P0;


for k = 1:nPairs
    Jk  = build_Jk(cd, c, L,  k);
    Jlk = build_Jk(cd, c, L,  L-k);
    Ps{k} = Jk + Jlk;
    Qs{k} = (Jk - Jlk)/1i;
end

Jmid     = build_Jk(cd, c, L, L/2);
Ps{L/2}  = Jmid + Jmid;


%%
%  Haux = 1 P1  + 0.1 P2 + 0.01 P3 + …  + (…) P0

eps  = 1.0;
Haux = eps * Ps{1};                 
for k = 2:numel(Ps)                 
    eps  = eps * 1e-1;
    Haux = Haux + eps * Ps{k};
end
for k = 1:numel(Qs)                 
    eps  = eps * 1e-1;
    Haux = Haux + eps * Qs{k};
end
eps  = eps * 1e-1;                  
Haux = Haux + eps * P0;

%%

[V,~] = eig(full(Haux));            
extract = @(A) diag(V' * A * V);    

Pvals = cell2mat(cellfun(extract, Ps ,'uni',0));
Qvals = cell2mat(cellfun(extract, Qs ,'uni',0));
Nvals = 0.5 * extract(P0);

%  print table
counter = (1:2^L).';
hdrP = sprintf('P%-2d ',1:size(Pvals,2));
hdrQ = sprintf('Q%-2d ',1:size(Qvals,2));
fprintf(['Rows: (Row# ',hdrP,hdrQ,'Ntot)\n']);
disp( real([counter, Pvals, Qvals, Nvals]) );

%%
% Cost Hamiltonian

Id   = speye(2^L);
Hobj = sparse(0);

for k = 1:numel(Ps)
    Hobj = Hobj + ( Ps{k} - Pvals(st,k)*Id )^2;
end
for k = 1:numel(Qs)
    Hobj = Hobj + ( Qs{k} - Qvals(st,k)*Id )^2;
end
Hobj = Hobj + ( Ntot - Nvals(st)*Id )^2;

n_angles = pi * rand(L,2);  
% project onto exact target state  |ψ_st⟩
fid_obj = V(:,st) * V(:,st)';

disp('----------------------------------------------------------------');
