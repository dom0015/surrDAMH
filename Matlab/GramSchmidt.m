function [ Q,R,imp ] = GramSchmidt( V )
%GRAMSCHMIDT_MODIF Summary of this function goes here
% modifikovany Gramuv-Schmidtuv proces (robustnejsi)
% po sloupcich
% prubezne ukladano do matic Q, R
% !!psano s durazem na jednoduchost kodu, ne efektivitu!!
[m,n]=size(V);
R=zeros(n);
Q=zeros(m,n);
imp=zeros(n,1);
temp=V(:,1);
R(1,1)=norm(temp);
Q(:,1)=temp/R(1,1);
for j=2:n
    temp=V(:,j);
    norm_orig=norm(temp);
    temp=temp/norm_orig;
    for k=1:j-1
        dotprod=temp'*Q(:,k);
        temp=temp-dotprod*Q(:,k);
        R(k,j)=dotprod*norm_orig;
    end
    norm_new=norm(temp);
    R(j,j)=norm_orig*norm_new;
    Q(:,j)=temp/norm_new;
    imp(j)=norm_new;
end

end

