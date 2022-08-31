function [U,V,memory_used_in_bytes] = mts2cca(X,Y,lx,ly,para)
    
% GraphNet  regression 
%
%  min_W  || W' X - Y||^2 + r1 * W*P*W + r2*||W||1
%  Input: (n: number of subjects, d: number of features)
%       - X: d x n
%       - Y: 1 x n
%       - P: d x d, Laplacian matrix of correlation
%       - r1, r2: tuned parameters
%  Output:
%       - W: d x 1 
%       - obj: results of each iteration
%  Author: Mansu Kim
%  Last update: 2018.10.08
[user, sys] = memory;
X = zscore(X);
Y = zscore(Y);
[~,n_feature_1] = size(X);
[~,n_feature_2,c] = size(Y);

U_old = ones(n_feature_1,c);
V_old = ones(n_feature_2,c);

XX = X'*X; % [p x p] = [p x n]*[n x p]
for i = 1:c
    XY(:,:,i) = X'*squeeze(Y(:,:,i)); % [p x q x c] = [p x n]*[n x q]
end
for i = 1:c
    YY(:,:,i) = squeeze(Y(:,:,i))'*squeeze(Y(:,:,i)); % [p x q x c] = [p x n]*[n x q]
end
d1 = ones(n_feature_1,1); 
d2 = ones(n_feature_2,1);

diff = 0.2;
U_diff = 1;
V_diff = 1; 
iter = 1;

beta1 = para(1);
beta2 = para(2);
lam1  = para(3);
lam2  = para(4);
r1    = para(5);
r2    = para(6);


while ( (U_diff > diff)|| (V_diff > diff))
    % fix v, solve u 
    
    % Compute U
    V = V_old;
    for i = 1:c
        XYV(:,i) = squeeze(XY(:,:,i))*V(:,i); % [p x c] = [p x q] * [q x c]
    end
    D1 = diag(d1);
    U  = ((1+ r1)*XX + lam1*lx + beta1*D1)\XYV;

    % normalization
    U = U./sqrt(diag(U'*XX*U))';
    
    % update D1
    d1 = 1./(2*(sqrt(sum(U.*U+eps,2))));
    
    % update U
    [u_m, u_idx] = max(abs(U_old(:)) - abs(U(:)));
    U_diff = max(u_m./(abs(U_old(u_idx))+eps));
    U_old = U;
    
    % fix u, solve v
    
    % Comput V
    D2 = diag(d2);
    for i = 1:c
        uXY = U(:,i)'*squeeze(XY(:,:,i));
        V(:,i) = ((1+r2)*YY(:,:,i)+lam2*ly(:,:,i)+beta2*D2)\uXY';
    end
    
    % normalization
    for i = 1:c
        V(:,i) = V(:,i)./sqrt(diag(V(:,i)'*YY(:,:,i)*V(:,i)))';
    end
    
    % update D2
    d2 = 1./(2*(sqrt(sum(V.*V+eps,2))));
    
    % update V
    [v_m, v_idx] = max(abs(V_old(:)) - abs(V(:)));
    V_diff = max(v_m./(abs(V_old(v_idx))+eps));
    V_old = V;
    
    for i = 1:c
        YV(:,i) = squeeze(Y(:,:,i))*V(:,i); %[p x c] = [p x q] * [q x c]
    end
    
    %obj(iter,:) = diag(corr(X*U,YV));
    obj = 1;
%     obj(iter,:) = trace((X*U-YV)'*(X*U-YV))+beta1*sum(sqrt(sum(U.*U,2)))+beta2*sum(sqrt(sum(V.*V,2)))+lam1*trace(U'*lx*U)+lam2*V(:,1)'*ly(:,:,1)*V(:,1)+lam2*V(:,2)'*ly(:,:,2)*V(:,2);
    iter = iter + 1;
    if iter > 400
        break
    end
end

[user2, sys2] = memory;
memory_used_in_bytes = user2.MemAvailableAllArrays - user.MemAvailableAllArrays
end
