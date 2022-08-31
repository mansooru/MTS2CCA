function [U,V,obj] = mts2cca(X,Y,lx,ly,para)
    
% Multi-task learning based structured sparse canonical correlation analysis
%
%  Input: (n: number of subjects, d: number of features)
%       - X: d x n
%       - Y: 1 x n
%       - lx, ly: d x d, Laplacian matrix of X and Y network structure
%       - para: parameters
%  Output:
%       - W: d x 1 
%       - obj: results of each iteration
%  Author: Mansu Kim

max_iter = 400;
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
    
    % objective function
    obj(iter,:) = trace((X*U-YV)'*(X*U-YV))+beta1*sum(sqrt(sum(U.*U,2)))+beta2*sum(sqrt(sum(V.*V,2)))+lam1*trace(U'*lx*U)+lam2*V(:,1)'*ly(:,:,1)*V(:,1)+lam2*V(:,2)'*ly(:,:,2)*V(:,2);
    diff_obj = abs(obj(iter)-obj(iter-1));
    iter = iter + 1;
    if iter > max_iter
        break
    end
    if diff_obj < err
        break
    end
end

end
