function [Z,S,P,F,E] = DLRRDP(X,F_ini,Z_ini,c,lambda1,lambda2,lambda3,max_iter)
    % The code is written by Zhiqiang Fu, 
    [m,n] = size(X);
    %% ---------- Initilization -------- %
    miu = 0.01;
    rho = 1.2;
    max_miu = 1e8;
    tol  = 1e-5;
    tol2 = 1e-2;
    zr   = 1e-9;
    C1 = zeros(m,n);
    C2 = zeros(n,n);
    E  = zeros(m,n);
    P = zeros(m,m);
    for iter = 1:max_iter
        if iter == 1
            Z = Z_ini;
            F = F_ini;
            S = Z_ini;
            clear Z_ini F_ini
        end
        S_old = S;
        P_old = P;
        Z_old = Z;
        E_old = E;
        %% -------- Update Z --------- %
        PX =P*X;
        Z = pinv((lambda2+miu)*eye(n)+miu*(PX'*PX))*miu*(PX'*(X-E+C1/miu)+S-C2/miu);
        Z = Z- diag(diag(Z));
        %% -------- Update S --------- %
        distX = L2_distance_1(PX,PX);
        distF = L2_distance_1(F',F');           
        dist  = distX+lambda1*distF;
        S     = Z+(C2-dist)/miu;
        S     = S - diag(diag(S));
        for ic = 1:n
            idx    = 1:n;
            idx(ic) = [];
            S(ic,idx) = EProjSimplex_new(S(ic,idx));          % 
        end
        %% ---------- Update F ----------- %
        LS = (S+S')/2;
        L = diag(sum(LS)) - LS;
        [F, ~, ev] = eig1(L, c, 0);
        %% update P
        Wz = (S+S')/2;
        Dz = diag(sum(Wz));
        Lz = Dz-Wz;
        M= X*Lz*X';
        XZ =X*Z;
        P = -miu*(E-X-C1/miu)*XZ'*pinv(eye(m)+4*M+miu*XZ*XZ');  
        %% ------- Update E ---------- %
        temp1 = X-P*X*Z+C1/miu;
        temp2 = lambda3/miu;
        E = max(0,temp1-temp2) + min(0,temp1+temp2);   
        %% -------- Update C1 C2 C3 miu -------- %
        L1 = X-P*X*Z-E;
        L2 = Z-S;
        C1 = C1+miu*L1;
        C2 = C2+miu*L2;
        LL1 = norm(Z-Z_old,'fro');
        LL2 = norm(S-S_old,'fro');
        LL3 = norm(P-P_old,'fro');
        LL4 = norm(E-E_old,'fro');
        SLSL = max(max(max(LL1,LL2),LL3),LL4)/norm(X,'fro');
        miu = min(rho*miu,max_miu);
        %% --------- obj ---------- %
        leq1 = max(max(abs(L1(:))),max(abs(L2(:))));
        stopC = max(leq1);
        if stopC < tol
            iter
            break;
        end   
    end
    end