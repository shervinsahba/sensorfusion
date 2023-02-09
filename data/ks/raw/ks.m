% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - nu*u_xxxx,  periodic BCs
% code revised by Shervin Sahba, 2022

clear all; clc; close all

rng(0)

SAVE_FIGS = false;

% parameters:
N = 2048;                   % 1-D grid size
L_x = 2*pi;                 % domain length
x = L_x/N*(1:N).';          % spatial grid
tmax = 10;                  % max value of time
h = 0.0001;                 % time step 
nu = 0.005;
nplt_factor = 1000;         % plotting factor
plot_all = true;            % overrides nplt and saves the solution at every timestep

% initial condition:
u = randn(N,1);             
% u = cos(x/16).*(1+sin(x/16));  % initial condition from Trefethen's paper, Fig 6.

% Precompute ETDRK4 scalar quantities: 
k = (2*pi/L_x)*fftshift(-N/2:N/2-1).';  % wave numbers 
g = -0.5i*k;
L = k.^2 - nu*k.^4;                     % Fourier multipliers u
E = exp(h*L); 
E2 = exp(h*L/2);
M = 16;                                 % no. of points for complex means 
r = exp(1i*pi*((1:M)-.5)/M);            % roots of unity 
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q =  h*real(mean((exp(LR/2)-1)./LR , 2));
f1 = h*real(mean((-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 , 2));
f2 = h*real(mean((2+LR+exp(LR).*(-2+LR))./LR.^3 , 2));
f3 = h*real(mean((-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 , 2));

% Main time-stepping loop:
nmax = round(tmax/h); 
nplt = floor((tmax/nplt_factor)/h); 
tt = zeros(1,nmax);
uu = zeros(N,nmax);
v = fft(u);
for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2); 
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;     
    if mod(n,nplt)==0 || plot_all 
        u = real(ifft(v));
        uu(:,n) = u; 
        tt(n) = t;
    end
end


%% Figures

% time cutoff for plots
% t_cutoff_max = tmax;
t_cutoff_max = 0.6;
t_cutoff = tt>0 & tt<=t_cutoff_max;
t_cutoff_tot = sum(t_cutoff);

% save(strcat('t_cutoff_tot','-',num2str(size(uu,1)),'x',num2str(size(uu,2))),'t_cutoff_tot')

h1 = figure(1);
% contour(x/(2*pi),tt(t_cutoff),uu(:,t_cutoff).',[-10 -5 0 5 10]),shading interp, colormap(hot)
% contour(x,tt,uu.'),shading interp, colormap(hot)
% surfl(x,tt(t_cutoff),uu(:,t_cutoff).'),shading interp, colormap(gray), view(15,50)
pcolor(x,tt(t_cutoff),uu(:,t_cutoff).'),shading interp, colormap('hot')
xlabel('x \in [0,2\pi]'); ylabel('t')
filename = strcat('ks','-',num2str(size(uu,1)),'x',num2str(size(uu,2)),'-frame','.png');
saveme(filename, SAVE_FIGS)

%%

for j=1:t_cutoff_tot
    uut(:,j)=abs(fft(uu(:,j)));
end
ks=fftshift(k);

h2 = figure(2);
surf(k,tt(t_cutoff),(uut.')),shading interp, view(15,50)
set(gca,'Xlim',[-50 50])
xlabel('k'); ylabel('t'); zlabel('|u_t|')
filename = strcat('ks','-',num2str(size(uu,1)),'x',num2str(size(uu,2)),'-spectrum','.png');
saveme(filename, SAVE_FIGS)

%% save data

% create a nice dataset, slicing off the transient behavior
uu_clean = uu(:,2001:end);
tt_clean = tt(2001:end);
tt_clean = tt_clean - tt_clean(1);

data = struct('uu',uu_clean,'tt',tt_clean);
save(strcat('ks-superset','-',num2str(size(uu_clean,1)),'x',num2str(size(uu_clean,2))),'data')


%% Animation loop (use 'MPEG-4' codec if available, else convert 'Motion JPEG AVI' to mp4 with ffmpeg)
ANIMATION_TRUE=false;
filename = strcat('ks','-',num2str(size(uu,1)),'x',num2str(size(uu,2)));
anim_length = floor(length(tt)/3);

if (length(tt)/t_cutoff_tot > 10) && ANIMATION_TRUE % sanity check. animation must have more than 10 frames!
    figure(99)
    video = VideoWriter(filename, 'Motion JPEG AVI'); video.Quality = 100;
    video.open()
    for t_anim = 1:anim_length
        pcolor(x,tt(t_anim:t_anim+t_cutoff_tot),uu(:,t_anim:t_anim+t_cutoff_tot).'),
            shading interp,colormap(hot)
        xlabel('x \in [0,2\pi]'); ylabel('t')
        title(['Kuramoto-Sivashinsky ',num2str(size(uu,1)),'x',num2str(size(uu,2))],'Interpreter','Latex') 
        drawnow
        video.writeVideo(getframe(gcf))
    end
    video.close()
else
    disp('warning: animation either disabled or canceled because it has less than 10 frames.')
end




%% Functions

function saveme(filename_, save_status)
if save_status
    saveas(gcf, filename_)
end
end
