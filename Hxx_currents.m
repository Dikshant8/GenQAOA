function [c, c_dagger] = create_fermionic_ops(L)
    
    sigma_z = sparse([1 2], [1 2], [1 -1], 2, 2);       % Sparse σ_z
    sigma_minus = sparse(2, 1, 1, 2, 2);                % Sparse σ^- 
    I = speye(2);                                       % Sparse id
    
    c = cell(1, L);
    c_dagger = cell(1, L);
    
    for j = 1:L
        ops = cell(1, L);
        for site = 1:L
            if site < j
                ops{site} = sigma_z;                    % Jordan-Wigner string
            elseif site == j
                ops{site} = sigma_minus;                
            else
                ops{site} = I;                          
            end
        end
        
        c{j} = ops{1};
        for site = 2:L
            c{j} = kron(c{j}, ops{site});               
        end
        c_dagger{j} = c{j}';                            
    end
end

function J_k = build_Jk(c_dagger, c, L, k)
    J_k = sparse(2^L, 2^L);
    for j = 1:L
        target_site = mod(j + k - 1, L) + 1;            % Periodic index
        
        J_k = J_k + sparse(c_dagger{j}*c{target_site});
    end
end

% Create fermionic operators
[c, c_dagger] = create_fermionic_ops(L);

% c: Cell array of annihilation operators 
% c_dagger: Cell array of creation operators



n_angles = pi * rand(L,2);  


%% build all operators

L=4;
fprintf('==== L = %d =====\n', L)

Ps = cell(floor(L/2),1);
Qs = Ps;

H0 = 0;
for k = 1:L/2
    J_k = build_Jk(c_dagger, c, L, k);
    Ps{k} = (J_k + J_k');
    Qs{k} = (J_k - J_k')/1i;
    H0 = H0 + 3^(-(k-1))*Ps{k} + 0*pi^(-(k-1)) * Qs{k};
end
testdiag = @(X) norm(X - diag(diag(X))) < 1e-12;

% find simultaneous eigenbasis
H0 = full(Ps{1}) + 0.1*Ps{2} + 0.01*Qs{1};

[V0, D0] = eig(H0);
eigPs = cell(size(Ps));
eigQs = cell(size(Qs));
for k = 1:L/2
    eigPs{k} = V0'*Ps{k}*V0;
    eigQs{k} = V0'*Qs{k}*V0;
    if ~testdiag(eigPs{k})
        fprintf('Warning: P_%d not diagonalized\n', k);
    else
        eigPs{k} = diag(eigPs{k});
    end
    if ~testdiag(eigQs{k})
        fprintf('Warning: Q_%d not diagonalized\n', k);
    else
        eigQs{k} = diag(eigQs{k});
    end
end

%%
Id = speye(2^L);

Hobj = (Ps{1}+0*Id)^2 + (Ps{2}+ 2*Id)^2 +  (Qs{1} -2*Id)^2;
fid_obj = V0(:,7);
fid_obj = fid_obj*fid_obj';

%%
real([eigPs{1}, eigPs{2}, eigQs{1}, eigQs{2}])
