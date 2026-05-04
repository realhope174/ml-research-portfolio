% MATLAB code goes here

function [spin] = Metropolis(spin, T, En, Mag, L, n, EngyMean, MagneticMean)
% METROPOLIS function performs the Metropolis algorithm for spin updates

global En;
global Mag;
global EngyMean;
global MagneticMean;
global En2Mean;
global Mag2Mean;
global Check;

j = 1;
E = 0;
M = 0;
En2 = 0;
Mag2 = 0;
y = 1;

for i = 1:n
    % Randomly select a spin to update
    linearIndex = randi(numel(spin));
    [row, col] = ind2sub(size(spin), linearIndex);

    % Calculate the energy change (dE) due to the spin flip
    N = Neighbor(L, row, col);
    dE = 2 * spin(row, col) * (spin(N(1), col) + spin(row, N(2)) + spin(row, N(3)) + spin(N(4), col));

    if dE >= 0
        prob = exp(-dE / T);
        x = rand();
        if x <= prob
            % Accept the spin flip
            spin(row, col) = -spin(row, col);
            En = En + dE;
            Mag = Mag + 2 * spin(row, col);
        end
    else
        spin(row, col) = -spin(row, col);
        En = En + dE;
        Mag = Mag + 2 * spin(row, col);
    end

    if (Check == 1)
        % Calculate quantities for checking purposes
        E = E + En / (L^2);
        M = M + Mag / (L^2);
        En2 = En2 + (En / (L^2))^2;
        Mag2 = Mag2 + (Mag / (L^2))^2;
    end

    if (T == 2) || (T == 2.5) || ((T == 1) && (Check == 0))
        if mod(i, 1000) == 0
            Energy(j) = En / (L^2);
            Magnetization(j) = Mag / (L^2);

            if ((mod(j, 1000) == 0) && (j <= 9000))
                if (T == 2)
                    figure(6)
                    sgtitle('Microscopic Configuration at T=2.0')
                else
                    figure(9)
                    sgtitle('Microscopic Configuration at T=2.5')
                end

                if (y <= 9)
                    subplot(3, 3, y);
                    image((spin + 1) * 100);
                    colormap(autumn);
                    xlabel(sprintf('Mag = %0.2f, En = %0.2f', Mag / L^2, En / L^2));
                    set(gca, 'YTickLabel', [], 'XTickLabel', []);
                    axis square;
                    colormap bone;
                    drawnow;
                    y = y + 1;
                else
                    y = 1;
                end
            end

            j = j + 1;
        end
    end
end

if (Check == 1)
    % Calculate mean values
    EngyMean = E / n;
    MagneticMean = M / n;
    En2Mean = En2 / n;
    Mag2Mean = Mag2 / n;
end

if (T == 1) && (Check == 0)
    Time = 1:1:j-1;
    Time = Time * 1000;

    figure(5)
    plot(Time, Magnetization, 'b')
    title('Magnetization at Equilibrium')
    xlabel('Time')
    ylabel('Magnetization')
end

if (T == 2) || (T == 2.5)
    Time = 1:1:j-1;
    Time = Time * 10000;

    figure(1)
    plot(Time, Energy)
    title('Energy')
    xlabel('Time')
    ylabel('Energy')
    hold on

    figure(2)
    plot(Time, Magnetization)
    title('Magnetization')
    xlabel('Time')
    ylabel('Magnetization')
    hold on
end
end
