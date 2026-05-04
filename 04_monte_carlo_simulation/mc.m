% Square-lattice 2D Ising model simulation

% Constants
kB = 1; % Boltzmann constant
J = 1; % Coupling constant
L = 100; % Linear size of the lattice
N = L^2; % Total number of spins
Tc = 2.269; % Critical temperature

% Functions

% Magnetization per spin
magnetization = @(sigma) sum(sigma(:)) / N;

% Energy per spin
energy = @(sigma) -J * sum(sum(sigma .* (...
    circshift(sigma, [1, 0]) + circshift(sigma, [-1, 0]) + ...
    circshift(sigma, [0, 1]) + circshift(sigma, [0, -1])))) / N;

% Mean magnetization as a function of temperature
meanMagnetization = @(T) (1 - (1 / sinh(2 * T * J))^2)^(1/8);

% Mean energy as a function of temperature
meanEnergy = @(T) -J * (1 + sinh(2 * T * J)^2) / 2;

% Magnetic susceptibility as a function of temperature
magneticSusceptibility = @(T) beta * N * ((1 - meanMagnetization(T)^2) / T);

% Specific heat as a function of temperature
specificHeat = @(T) beta^2 * N * ((meanEnergy(T)^2 - meanEnergy(T^2)) / T^2);

% Initial configurations
initialConfigurations = zeros(L, L, 3);
initialConfigurations(:, :, 1) = ones(L);
initialConfigurations(:, :, 2) = -ones(L);
initialConfigurations(:, :, 3) = sign(rand(L) - 0.5);

% Compute observables

% Magnetization and energy as a function of time
T1 = 2; % Temperature below critical temperature
T2 = 2.5; % Temperature above critical temperature
numIterations = 100; % Number of iterations for magnetization and energy computation

% Initialize arrays to store magnetization and energy over time
magnetizationTime = zeros(numIterations, 2);
energyTime = zeros(numIterations, 2);

% Perform simulations for T1 and T2
for k = 1:3
    sigma = initialConfigurations(:, :, k);
    
    for i = 1:numIterations
        magnetizationTime(i, 1) = magnetization(sigma);
        energyTime(i, 1) = energy(sigma);
        
        % Perform Monte Carlo update
        for sweep = 1:N
            i = randi(L);
            j = randi(L);
            deltaE = 2 * J * sigma(i, j) * (...
                sigma(mod(i-2, L)+1, j) + sigma(mod(i, L)+1, j) + ...
                sigma(i, mod(j-2, L)+1) + sigma(i, mod(j, L)+1));
            if deltaE <= 0 || exp(-deltaE / (kB * T1)) > rand()
                sigma(i, j) = -sigma(i, j);
            end
        end
    end
    
    % Store final configuration
    initialConfigurations(:, :, k) = sigma;
    
    % Plot magnetization as a function of time
    figure;
    plot(1:numIterations, magnetizationTime(:, 1));
    xlabel('Time');
    ylabel('Magnetization');
    title(['Magnetization as a function of time (T = 2), Initial Configuration = ' num2str(k)]);
    
    % Plot energy as a function of time
    figure;
    plot(1:numIterations, energyTime(:, 1));
    xlabel('Time');
    ylabel('Energy');
    title(['Energy as a function of time (T = 2), Initial Configuration = ' num2str(k)]);
end

% Compute observables as functions of temperature

% Temperature range
T = linspace(1.5, 3, 100);

% Initialize arrays to store observables
meanMagnetizationValues = zeros(size(T));
meanEnergyValues = zeros(size(T));
magneticSusceptibilityValues = zeros(size(T));
specificHeatValues = zeros(size(T));

% Perform simulations for each temperature
for i = 1:length(T)
    magnetizations = zeros(1, 1000);
    energies = zeros(1, 1000);
    
    for k = 1:1000
        sigma = sign(rand(L) - 0.5);
        
        for sweep = 1:N
            i = randi(L);
            j = randi(L);
            deltaE = 2 * J * sigma(i, j) * (...
                sigma(mod(i-2, L)+1, j) + sigma(mod(i, L)+1, j) + ...
                sigma(i, mod(j-2, L)+1) + sigma(i, mod(j, L)+1));
            if deltaE <= 0 || exp(-deltaE / (kB * T(i))) > rand()
                sigma(i, j) = -sigma(i, j);
            end
        end
        
        magnetizations(k) = magnetization(sigma);
        energies(k) = energy(sigma);
    end
    
    meanMagnetizationValues(i) = mean(magnetizations);
    meanEnergyValues(i) = mean(energies);
    magneticSusceptibilityValues(i) = magneticSusceptibility(T(i));
    specificHeatValues(i) = specificHeat(T(i));
end

% Plot mean magnetization as a function of temperature
figure;
plot(T, meanMagnetizationValues);
xlabel('Temperature');
ylabel('Mean Magnetization');
title('Mean Magnetization as a function of Temperature');

% Plot mean energy as a function of temperature
figure;
plot(T, meanEnergyValues);
xlabel('Temperature');
ylabel('Mean Energy');
title('Mean Energy as a function of Temperature');

% Plot magnetic susceptibility as a function of temperature
figure;
plot(T, magneticSusceptibilityValues);
xlabel('Temperature');
ylabel('Magnetic Susceptibility');
title('Magnetic Susceptibility as a function of Temperature');

% Plot specific heat as a function of temperature
figure;
plot(T, specificHeatValues);
xlabel('Temperature');
ylabel('Specific Heat');
title('Specific Heat as a function of Temperature');

% Sampled microscopic configurations

% Time points
t = 2e3 * (1:9);

% Initialize array to store configurations
samplingConfigurations = zeros(L, L, 9);

% Perform simulations for T1 and T2
for k = 1:9
    sigma = initialConfigurations(:, :, k);
    
    for i = 1:t(k)
        % Perform Monte Carlo update
        for sweep = 1:N
            i = randi(L);
            j = randi(L);
            deltaE = 2 * J * sigma(i, j) * (...
                sigma(mod(i-2, L)+1, j) + sigma(mod(i, L)+1, j) + ...
                sigma(i, mod(j-2, L)+1) + sigma(i, mod(j, L)+1));
            if deltaE <= 0 || exp(-deltaE / (kB * T1)) > rand()
                sigma(i, j) = -sigma(i, j);
            end
        end
    end
    
    % Store final configuration
    samplingConfigurations(:, :, k) = sigma;
    
    % Plot configuration
    figure;
    imagesc(sigma);
    colormap([-1 -1 -1; 1 0 0; 0 0 1]);
    axis square;
    title(['Microscopic Configuration, t = ' num2str(t(k)) ', Initial Configuration = ' num2str(k)]);
end
