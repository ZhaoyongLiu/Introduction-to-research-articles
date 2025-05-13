%% 
% A Continuous-Time Observer Which Converges in Finite Time,Robert Engel,TAC,2002.
% Author         Date         Version    Modification
% Zhaoyong Liu   May-13-2025  1.0

clc; clear; close all;

A = [0 2; -1 0]; B = [0; 1]; C = [1 0];
sigma = -1;

poles1 = [ -1+sigma, -2+sigma];  % Desired observer poles
poles2 = [ 0.1+sigma, 0.2+sigma];

% Check the observability of the system

if rank (obsv(A,C)) == size(A,1)
   disp('The system is observable!' );
else
   error ('The system is not observable!' );
end

H1 = place(A', C', poles1)';  % Compute the observer gain using pole placement
H2 = place(A', C', poles2)';

t0 = 0;
tf = 20;
tspan = [t0, tf];

x0 = [10; -5];   % System initial condition
z10 = [0 ; 0];   % Observer initial condition
z20 = [0 ; 0];
initValues = {x0, z10, z20};
Matrices = {A, B, C, H1, H2};

D = 1;   % finite time to estimate the state
dimn = size(A,1);
F1 = A - H1*C;  F2 = A - H2*C;
F = [F1 zeros(dimn); zeros(dimn) F2];
T = [eye(dimn); eye(dimn)];

if det([T, expm(F*D)*T]) ~= 0
   disp('det[T, exp(FD)*T] ~= 0 !' );
else
   error ('det[T, exp(FD)*T] = 0 !' );
end

%% Numerical simulation
h = 0.001;      % step size

[t,x,u,y,z,zDelay,xhat] = Euler(@system,@observer,tspan,h,D,initValues,Matrices);


%% Plot figures
figure(1)
subplot(2,1,1)
plot(t,x(1,:),'k', t, xhat(1,:),'b--','LineWidth',1.2);
legend('$x_1(t)$','$\hat{x}_1(t)$','Interpreter','Latex','Fontsize',13);
axis([t0 tf -inf inf])

subplot(2,1,2)
plot(t,x(2,:),'k', t, xhat(2,:),'r-.','LineWidth',1.2);
legend('$x_2(t)$','$\hat{x}_2(t)$','Interpreter','Latex','Fontsize',13);
axis([t0 tf -inf inf])
xlabel('Time (s)')

figure(2)
subplot(2,1,1)
plot(t,x(1,:)-xhat(1,:),'k','LineWidth',1.2);
min1 = min(x(1,:)-xhat(1,:));
max1 = max(x(1,:)-xhat(1,:));
line([D D], [min1 max1],'LineStyle','-.','Color','m','LineWidth',1.2)
legend('$x_1(t)-\hat{x}_1(t)$','$t=D$','Interpreter','Latex','Fontsize',13);
axis([t0 tf -inf inf])
grid on;

subplot(2,1,2)
plot(t,x(2,:)-xhat(2,:),'k','LineWidth',1.2);
min2 = min(x(2,:)-xhat(2,:));
max2 = max(x(2,:)-xhat(2,:));
line([D D], [min2 max2+1],'LineStyle','-.','Color','m','LineWidth',1.2)
legend('$x_2(t)-\hat{x}_2(t)$','$t=D$','Interpreter','Latex','Fontsize',13);
axis([t0 tf -inf max2+1])
grid on;
xlabel('Time (s)')

figure(3)
subplot(2,1,1)
plot(t,z(1,:),'k', t, zDelay(1,:),'--','LineWidth',1.2);
legend('$z^1(t)$','$z^1(t-D)$','Interpreter','Latex','Fontsize',13);
axis([t0 tf -inf inf])

subplot(2,1,2)
plot(t,z(2,:),'k', t, zDelay(2,:),'b--','LineWidth',1.2);
legend('$z^2(t)$','$z^2(t-D)$','Interpreter','Latex','Fontsize',13);
axis([t0 tf -inf inf])
xlabel('Time (s)')

%% Forward euler method for solving ode
function [t,x,u,y,z,zDelay,xhat] = Euler(system,observer,tspan,h,D,initValues,Matrices)
   
   t0 = tspan(1);
   tf = tspan(2);
   n = floor((tf-t0)/h);

   x0 = initValues{1};
   z10 = initValues{2};
   z20 = initValues{3};

   A = Matrices{1}; B = Matrices{2}; C = Matrices{3};
   H1 = Matrices{4}; H2 = Matrices{5};

   dimn = size(A,1); dimm = size(B,2); dimp = size(C,1);

   F1 = A - H1*C;  F2 = A - H2*C;
   F = [F1 zeros(dimn); zeros(dimn) F2];
   T = [eye(dimn); eye(dimn)]; 
   K = [eye(dimn), zeros(dimn)]*inv([T, expm(F*D)*T]);

   t = zeros(1,n+1);  t(1) = t0;  
   x = zeros(dimn,n+1);  x(:,1) = x0; 
   z1 = zeros(dimn,n+1);  z1(:,1) = z10; 
   z2 = zeros(dimn,n+1);  z2(:,1) = z20;
   z = [z1; z2];  
   zDelay = zeros(dimn*2,n+1);

   xhat = zeros(dimn,n+1); xhat(:,1) = K*(z(:,1)-expm(F*D)*zDelay(:,1));

   u = zeros(dimm, n+1);  u(dimm,1) = 1;
   y = zeros(dimp, n+1);  y(dimp,1) = C*x0;

   for i = 1:n
       t(i+1) = t(i)+h;
       u(i+1) = 1;

       dotx = system(t(i), x(:,i), u(:,i), A, B);
       x(:,i+1) = x(:, i) + h*dotx;

       y(:,i+1) = C*x(:,i+1);

       dotz1 = observer(t(i), z1(:,i), u(:,i), y(:,i), A, B, C, H1);
       z1(:,i+1) = z1(:,i) + h*dotz1;

       dotz2 = observer(t(i), z2(:,i), u(:,i), y(:,i), A, B, C, H2);
       z2(:,i+1) = z2(:,i) + h*dotz2;

       z(:,i+1) = [z1(:,i+1); z2(:,i+1)];

       if t(i+1) >= D
          index = floor(D/h);
          zDelay(:,i+1) = z(:,i+1-index);
       end
       
       xhat(:,i+1) = K*(z(:,i+1)-expm(F*D)*zDelay(:,i+1));

   end

end

%% System and observer dynamics
function dotx = system(t,x,u,A,B)
    
    dotx = A*x + B*u;
end

function dotzi = observer(t,zi,u,y,A,B,C,Hi)
    
    dotzi = (A-Hi*C)*zi + Hi*y +B*u;
end

