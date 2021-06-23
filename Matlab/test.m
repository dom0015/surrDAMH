ii = csvread("ii.csv")+1;
jj = csvread("jj.csv")+1;
vv = csvread("vv.csv");
solutions = csvread("solutions.csv");
b = csvread("b.csv");

A=sparse(ii,jj,vv);


cg_accuracy=1e-6;
max_iter=1000;


W=solutions(:,1); W=W/norm(W);

N_sam=100;

%% orthogonalization of deflation basis
W=solutions;
d=svd(W); disp(min(d))
[Q,R,imp]=GramSchmidt(W);

L=ichol(A);
%[x,flag,relres,iter,resvec]=pcg(Af,bf,1e-8,10000,L*L');
precond=@(x)L\((L')\x);

%% observation 1 (using PDCG)
iterations=zeros(1,N_sam);
residuals=zeros(1,N_sam);
times=zeros(1,N_sam);
times_without=zeros(1,N_sam);
for n=0:N_sam-1
    tic;
    [x,iter,resvec_dcg,tag,t_wo] = PDCG( A,b,[],Q(:,1:n),[],precond,cg_accuracy,max_iter);
    t=toc;
    iterations(n+1)=iter;
    residuals(n+1)=resvec_dcg(end);
    times(n+1)=t;
    times_without(n+1)=t_wo;
end

figure; plot(iterations); grid on
figure; plot(residuals); grid on
figure; plot(times); hold on; plot(times_without); grid on
