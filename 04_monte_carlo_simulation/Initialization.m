function [spin, Mag] = Initialization(L, IC)
% INITIALIZATION ALGORITHM
if IC == 1
    spin = ones(L, L);
elseif IC == 2
    spin = -ones(L, L);
else
    spin = sign(0.5 - rand(L, L));
end

global Mag;
global En;
global EngyMean;
global MagneticMean;
En = 0;
Mag = 0;

for x = 1:L
    for y = 1:L
        N = Neighbor(L, x, y);
        Mag = Mag + spin(x, y); % Initial magnetization
        En = En - spin(x, y) * (spin(N(4), y) + spin(x, N(2))); % Initial energy
    end
end

En
Mag
EngyMean = En / L^2
MagneticMean = Mag / L^2
end
