% Parameters
L = 100;                      % Linear size of the lattice
N = L^2;                      % Total number of spins
k_B = 1;                      % Boltzmann constant
T_values = [2, 2.5];          % Temperatures to simulate
num_iterations = 2e3;         % Number of iterations/sweeps
num_saved_configurations = 9; % Number of saved microscopic configurations

% Initialize lattice
sigma = ones(L, L);           % Initialize all spins to +1
% OR
% sigma = -ones(L, L);        % Initialize all spins to -1
% OR
% sigma = sign(rand(L, L) - 0.5);  % Initialize spins randomly

% Preallocate arrays for observables
magnetization = zeros(num_iterations, numel(T_values));
energy = zeros(num_iterations, numel(T_values));
mean_magnetization = zeros(numel(T_values), 1);
mean_energy = zeros(numel(T_values), 1);
magnetization_sq = zeros(numel(T_values), 1);
energy_sq = zeros(numel(T_values), 1);

% Preallocate array for microscopic configurations
saved_configurations = zeros(L, L, num_saved_configurations);

% Monte Carlo simulation
for t = 1:num_iterations
    for idx_T = 1:numel(T_values)
        T = T_values(idx_T);

        % Perform single Monte Carlo step for each spin
        for i = 1:N
            % Randomly select a lattice site
            row = randi(L);
            col = randi(L);

            % Calculate the change in energy if the spin is flipped
            deltaE = 2 * sigma(row, col) * ...
                (sigma(mod(row - 2, L) + 1, col) + sigma(row, mod(col - 2, L) + 1) + ...
                sigma(mod(row, L) + 1, col) + sigma(row, mod(col, L) + 1));

            % Accept or reject the new spin configuration
            if deltaE <= 0 || rand < exp(-deltaE / (k_B * T))
                sigma(row, col) = -sigma(row, col);
            end
        end

        % Compute observables
        magnetization(t, idx_T) = sum(sigma(:)) / N;
        energy(t, idx_T) = -sum(sum(sigma .* (circshift(sigma, [1, 0]) + circshift(sigma, [-1, 0]) + ...
            circshift(sigma, [0, 1]) + circshift(sigma, [0, -1])))) / N;

        % Accumulate values for mean magnetization and energy
        mean_magnetization(idx_T) = mean_magnetization(idx_T) + magnetization(t, idx_T);
        mean_energy(idx_T) = mean_energy(idx_T) + energy(t, idx_T);
        magnetization_sq(idx_T) = magnetization_sq(idx_T) + magnetization(t, idx_T)^2;
        energy_sq(idx_T) = energy_sq(idx_T) + energy(t, idx_T)^2;

        % Save microscopic configurations at specific times
        if ismember(t, round(linspace(1, num_iterations, num_saved_configurations)))
saved_configurations(:, :, find(ismember(round(linspace(1, num_iterations, num_saved_configurations)), t))) = sigma;
end
end
end

% Compute mean values and other observables
mean_magnetization = mean_magnetization / num_iterations;
mean_energy = mean_energy / num_iterations;
magnetization_sq = magnetization_sq / num_iterations;
energy_sq = energy_sq / num_iterations;

% Calculate magnetic susceptibility and specific heat
magnetic_susceptibility = (magnetization_sq - mean_magnetization.^2) ./ (k_B * T_values);
specific_heat = (energy_sq - mean_energy.^2) ./ (k_B * T_values.^2);

% Plotting Observables
figure;
subplot(2, 2, 1);
plot(1:num_iterations, magnetization(:, 1), 'b', 'LineWidth', 1.5);
hold on;
plot(1:num_iterations, magnetization(:, 2), 'r', 'LineWidth', 1.5);
xlabel('Time');
ylabel('Magnetization');
legend('T = 2', 'T = 2.5');

subplot(2, 2, 2);
plot(1:num_iterations, energy(:, 1), 'b', 'LineWidth', 1.5);
hold on;
plot(1:num_iterations, energy(:, 2), 'r', 'LineWidth', 1.5);
xlabel('Time');
ylabel('Energy');
legend('T = 2', 'T = 2.5');

subplot(2, 2, 3);
plot(T_values, mean_magnetization, 'ko-', 'LineWidth', 1.5);
xlabel('Temperature');
ylabel('Mean Magnetization');

figure;
for idx_T = 1:numel(T_values)
    T = T_values(idx_T);
    
    subplot(1, numel(T_values), idx_T);
    imagesc(sigma);
    colormap([0 0 1; 1 1 1; 1 0 0]);
    axis square;
    title(['T = ' num2str(T)]);
end

subplot(2, 2, 4);
plot(T_values, mean_energy, 'ko-', 'LineWidth', 1.5);
xlabel('Temperature');
ylabel('Mean Energy');

% Plot magnetic susceptibility and specific heat
figure;
subplot(1, 2, 1);
plot(T_values, magnetic_susceptibility, 'ko-', 'LineWidth', 1.5);
xlabel('Temperature');
ylabel('Magnetic Susceptibility');

subplot(1, 2, 2);
plot(T_values, specific_heat, 'ko-', 'LineWidth', 1.5);
xlabel('Temperature');
ylabel('Specific Heat');

% Plot saved microscopic configurations
figure;
for i = 1:num_saved_configurations
subplot(3, 3, i);
imagesc(saved_configurations(:, :, i));
colormap([0 0 1; 1 1 1; 1 0 0]);
axis square;
title(['Time = ' num2str(round(linspace(1, num_iterations, num_saved_configurations), i))]);
end