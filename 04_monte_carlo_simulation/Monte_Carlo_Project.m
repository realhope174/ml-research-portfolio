clear; clc; close all;
tic
global MagneticMean;
global EngyMean;
global Mag;
global En;
global En2Mean;
global Mag2Mean;
global Check;

L = input("Enter Lattice Size: ");
IC = input("Enter The initial conditions \n1 = Positive, 2 = Negative, 3 = Random\n");
Check = 0;
spin = Initialization(L, IC);
n = 10^8;

spin = Metropolis(spin, 1, En, Mag, L, n, MagneticMean, EngyMean);
n = 10^7;
i = 1;

for T = 1:0.1:4
    Check = 1;
    spin = Metropolis(spin, T, En, Mag, L, n);
    Energy(i) = EngyMean;
    Magnetization(i) = MagneticMean;
    MagSus(i) = ((L^2) / T) * (Mag2Mean - (MagneticMean^2));
    SpesHeat(i) = ((L / T)^2) * (En2Mean - (EngyMean^2));
    i = i + 1;
end

Temp = 1:0.1:4;
figure(3) % FIGURE 3
plot(Temp, Energy, 'b-o')
title('Mean Energy')
xlabel('Temperature')
ylabel('Energy')

figure(4) % FIGURE 4
plot(Temp, Magnetization, 'r-o')
title('Mean Magnetization')
xlabel('Temperature')
ylabel('Magnetization')

figure(7) % FIGURE 7
plot(Temp, SpesHeat, 'm*')
title('Specific Heat')
xlabel('Temperature')
ylabel('Specific Heat')
SpesHeat

figure(8) % FIGURE 8
plot(Temp, MagSus, 'm*')
title('Magnetic Susceptibility')
xlabel('Temperature')
ylabel('Magnetic Susceptibility')
MagSus

toc
