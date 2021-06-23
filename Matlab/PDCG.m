function [ x,iter,resvec,tag,time] = PDCG( A,b,x0,W,Q,M,tol,maxiter)
%DPCG Summary of this function goes here
%   Detailed explanation goes her
if isempty(W)
   W=zeros(size(b)); 
   Q=@(x)0;
end
if isempty(Q)
    if ~isa(A, 'numeric')
        disp('If A is not a matrix, provide Q!');
        return 
    end
    WTAW=W'*A*W;
    WTA=W'*A;
    Q=@(x)W*(WTAW\((W')*x));
    P=@(x)x-W*(WTAW\(WTA*x));
else
    P=@(x)x;
end
if isa(A, 'numeric')
    A_mat=A;
    A=@(x)A_mat*x;
end
if isempty(M)
    M=@(x)x;
end
if isa(M, 'numeric')
    M_mat=M;
    M=@(x)M_mat\x;
end
if isempty(x0)
    x0=0*b;%W*Q(W'*b);
end

%P=@(x)x-Q(A(x));
tic;
r0=b-A(x0);
x=x0+Q(r0);
b_norm=norm(b);
res=norm(A(x)-b)/b_norm;
if res<tol || maxiter==0
    tag=0;
%     disp([tag size(W,2) 0])
    resvec=res;
    iter=0;
    return
end
r=b-A(x);
z=M(r);
p=P(z);

gamma_old=dot(r,z);
tag=3;
resvec=zeros(maxiter+1,1);
resvec(1)=res;
for j=1:maxiter
    s=A(p);
    %s=P(A(p));
    alpha=gamma_old/dot(s,p);
    x=x+alpha*p;
    res=norm(A(x)-b);%/b_norm;
    resvec(j+1)=res;
    if res<tol
        tag=1;
        break;
    end
%     if res>2*min(resvec(1:j))
%         tag=2;
%         break;
%     end
    r=r-alpha*s;
    z=M(r);
    gamma_new=dot(r,z);
    beta=gamma_new/gamma_old;
    p=P(z)+beta*p;
    %p=z+beta*p;
    gamma_old=gamma_new;
end

resvec=resvec(1:j+1);
iter=j;
%disp([tag size(W,2) j])
time=toc;
end