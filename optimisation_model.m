%% ---------- Load Returns data ----------
portfoliocomponents = char({'Cash', 'RealEstate', 'Equity', 'Gold', 'Tbill'});
returnsTable = readtable('Compiled_Returns_Data.csv');
returns = table2array(returnsTable(:,1:5));

% ---------- End of Load Returns data ----------

%% ---------- Constants ----------
% Expected returns, risk free asset returns and asset covariance matrix
global expReturns E_risk_free covMatrix;
expReturns = [0.002, 0.096, 0.113, 0.106, 0.03];
E_risk_free = 0.03;
covMatrix = cov(returns);

% Risk-aversion factor
global alpha
alpha = 0.5;

% Constraint 2: Maximum volatility
sigma_p_max = 0.1575;

% ---------- End of Constants ----------

%% ---------- Objective function against weightage ----------
syms y1 y3 y4 y5
assume(y1, "real")
assume(y3, "real")
assume(y4, "real")
assume(y5, "real")

y2 = 1 - y1 - y3 - y4 - y5;
weights = [y1; y2; y3; y4; y5];

[E_p,sigma_p] = calc(weights,expReturns,covMatrix);
[~,~,weighted_U_sym] = obj(E_p,sigma_p,E_risk_free,alpha);

y_values = linspace(0.1, 0.6, 500);

% Substitute numeric values into the derivatives
Df_1Numeric = vpa(subs(weighted_U_sym, ...
    [y3; y4; y5], ...
    [0.1; 0.1; 0.1])) ;

Df_1_func = matlabFunction(Df_1Numeric);
Df_1_numbers = Df_1_func(y_values);

Df_3Numeric = vpa(subs(weighted_U_sym, ...
    [y1; y4; y5], ...
     [0.1; 0.1; 0.1])) ;

Df_3_func = matlabFunction(Df_3Numeric);
Df_3_numbers = Df_3_func(y_values);

Df_4Numeric = vpa(subs(weighted_U_sym, ...
    [y1; y3; y5], ...
     [0.1; 0.1; 0.1])) ;

Df_4_func = matlabFunction(Df_4Numeric);
Df_4_numbers = Df_4_func(y_values);

Df_5Numeric = vpa(subs(weighted_U_sym, ...
    [y1; y3; y4], ...
     [0.1; 0.1; 0.1])) ;

Df_5_func = matlabFunction(Df_5Numeric);
Df_5_numbers = Df_5_func(y_values);

% Visualisation
figure1 = figure('Name','Objective Function against Weightage');

subplot(2, 2, 1)
plot(y_values, Df_1_numbers)
xlabel('Weightage') 
ylabel('Objective Function') 
title('Objective function value with respect to proportion of cash')

subplot(2, 2, 2)
plot(y_values, Df_3_numbers)
xlabel('Weightage') 
ylabel('Objective Function') 
title('Objective function value with respect to proportion of equity')

subplot(2, 2, 3)
plot(y_values, Df_4_numbers)
xlabel('Weightage') 
ylabel('Objective Function')  
title('Objective function value with respect to proportion of gold')

subplot(2, 2, 4)
plot(y_values, Df_5_numbers)
xlabel('Weightage') 
ylabel('Objective Function')   
title('Objective function value with respect to proportion of Tbill')

% ---------- End of Objective function against weightage ----------

%% ---------- Monotonicity ----------
syms y1 y3 y4 y5
assume(y1, "real")
assume(y3, "real")
assume(y4, "real")
assume(y5, "real")

y2 = 1 - y1 - y3 - y4 - y5;
weights = [y1; y2; y3; y4; y5];

[E_p,sigma_p] = calc(weights,expReturns,covMatrix);
[~,~,weighted_U_sym] = obj(E_p,sigma_p,E_risk_free,alpha);

Df_1 = vpa(diff(weighted_U_sym, y1));
Df_3 = vpa(diff(weighted_U_sym, y3));
Df_4 = vpa(diff(weighted_U_sym, y4));
Df_5 = vpa(diff(weighted_U_sym, y5));

y_values = linspace(0.1, 0.2, 500);

% Substitute numeric values into the derivatives
Df_1Numeric = vpa(subs(Df_1, ...
    [y3; y4; y5], ...
    [0.1; 0.1; 0.1])) ;

Df_1_func = matlabFunction(Df_1Numeric);
Df_1_numbers = Df_1_func(y_values);

Df_3Numeric = vpa(subs(Df_3, ...
    [y1; y4; y5], ...
    [0.2; 0.2; 0.2])) ;

Df_3_func = matlabFunction(Df_3Numeric);
Df_3_numbers = Df_3_func(y_values);

Df_4Numeric = vpa(subs(Df_4, ...
    [y1; y3; y5], ...
    [0.2; 0.2; 0.2])) ;

Df_4_func = matlabFunction(Df_4Numeric);
Df_4_numbers = Df_4_func(y_values);

Df_5Numeric = vpa(subs(Df_5, ...
    [y1; y3; y4], ...
    [0.2; 0.2; 0.2])) ;

Df_5_func = matlabFunction(Df_5Numeric);
Df_5_numbers = Df_5_func(y_values);

% Visualisation
figure2 = figure('Name','Monotonicity');

subplot(2, 2, 1)
plot(y_values, Df_1_numbers)
xlabel('Proportion of Cash') 
ylabel('Gradient') 
title('Gradient with respect to proportion of cash')

subplot(2, 2, 2)
plot(y_values, Df_3_numbers)
xlabel('Proportion of Equity') 
ylabel('Gradient') 
title('Gradient with respect to proportion of equity')

subplot(2, 2, 3)
plot(y_values, Df_4_numbers)
xlabel('Proportion of Gold') 
ylabel('Gradient') 
title('Gradient with respect to proportion of gold')

subplot(2, 2, 4)
plot(y_values, Df_5_numbers)
xlabel('Proportion of Tbill') 
ylabel('Gradient') 
title('Gradient with respect to proportion of Tbill')

% ---------- End of Monotonicity ----------

%% ---------- Design sensitivities ----------
syms z1 z3 z4 z5
assume(z1, "real")
assume(z3, "real")
assume(z4, "real")
assume(z5, "real")

z2 = 1 - z1 - z3 - z4 - z5;
weights = [z1; z2; z3; z4; z5];

[E_p,sigma_p] = calc(weights,expReturns,covMatrix);
[S_p_sym,U_sym,weighted_U_sym] = obj(E_p,sigma_p,E_risk_free,alpha);

Df_1 = vpa((z1/weighted_U_sym)*diff(weighted_U_sym, z1));
Df_3 = vpa((z3/weighted_U_sym)*diff(weighted_U_sym, z3));
Df_4 = vpa((z4/weighted_U_sym)*diff(weighted_U_sym, z4));
Df_5 = vpa((z5/weighted_U_sym)*diff(weighted_U_sym, z5));

z_values = linspace(0.1, 0.2, 500);

% Substitute numeric values into the derivatives
Df_1Numeric = vpa(subs(Df_1, ...
    [z3; z4; z5], ...
    [0.2; 0.2; 0.2])) ;

Df_1_func = matlabFunction(Df_1Numeric);
Df_1_numbers = Df_1_func(z_values);

Df_3Numeric = vpa(subs(Df_3, ...
    [z1; z4; z5], ...
    [0.2 ;0.2; 0.2])) ;

Df_3_func = matlabFunction(Df_3Numeric);
Df_3_numbers = Df_3_func(z_values);

Df_4Numeric = vpa(subs(Df_4, ...
    [z1; z3; z5], ...
    [0.2; 0.2; 0.2])) ;

Df_4_func = matlabFunction(Df_4Numeric);
Df_4_numbers = Df_4_func(z_values);

Df_5Numeric = vpa(subs(Df_5, ...
    [z1; z3; z4], ...
    [0.2; 0.2; 0.2])) ;

Df_5_func = matlabFunction(Df_5Numeric);
Df_5_numbers = Df_5_func(z_values);

% Visualisation
figure3 = figure('Name','Design Sensitivities');

subplot(2, 2, 1)
plot(z_values, Df_1_numbers)
xlabel('Proportion of Cash') 
ylabel('Log Gradient') 
title('Log Gradient with respect to proportion of cash')

subplot(2, 2, 2)
plot(z_values, Df_3_numbers)
xlabel('Proportion of Equity') 
ylabel('Log Gradient') 
title('Log Gradient with respect to proportion of equity')

subplot(2, 2, 3)
plot(z_values, Df_4_numbers)
xlabel('Proportion of Gold') 
ylabel('Log Gradient') 
title('Log Gradient with respect to proportion of gold')

subplot(2, 2, 4)
plot(z_values, Df_5_numbers)
xlabel('Proportion of Tbill') 
ylabel('Log Gradient') 
title('Log Gradient with respect to proportion of Tbill')

% ---------- End of Design Sensitivities ----------

%% ---------- Boundedness Check ----------
number_of_assets = size(returns, 2);

number_of_portfolios = 1000000; 
results = zeros(3, number_of_portfolios);
weights_matrix = zeros(number_of_portfolios, number_of_assets);

for i = 1:number_of_portfolios
    weights = rand(number_of_assets, 1);
    weights = weights/sum(weights);

    [E_P,sigma_p] = calc(weights,expReturns,covMatrix);
    [S_p,U,weighted_U] = obj(E_P,sigma_p,E_risk_free,alpha);

    results(1, i) = E_P;
    results(2, i) = sigma_p;
    results(3, i) = weighted_U;

    weights_matrix(i, :) = weights';
end

% Visualisation
figure3 = figure('Name','Boundedness Check');
scatter(results(2,:),results(3,:),50,"red")
xlabel('Volatility')
ylabel('Objective function')

%[minU,index]=min(results(3,:));
%corrweights=weights_matrix(index,:);

% ---------- End of Boundedness Check ----------

%% ---------- Convexity ----------
syms x1 x2 x3 x4 x5

weights = [x1; x2; x3; x4; x5];

[E_p_sym,sigma_p_sym] = calc(weights,expReturns,covMatrix);
[S_p_sym,U_sym,weighted_U_sym] = obj(E_p_sym,sigma_p_sym,E_risk_free,alpha);

hessianMatrix = hessian(weighted_U_sym, weights);

weight = [0.2;0.2;0.2;0.2;0.2];

% Substitute numeric values into the Hessian matrix
HessianMatrixNumeric = vpa(subs(hessianMatrix, ...
    [weights(:)], ...
    [weight(:)])) ;

eigenValues = eig(HessianMatrixNumeric);

% If all eigenvalues are non-negative, the function is convex
isConvex = all(eigenValues >= 0);

if isConvex
    disp('The objective function is convex.');
else
    disp('The objective function is not convex.');
end

% ---------- End of Convexity ----------

%% ---------- Markowitz Efficient Frontier ----------
returns = returns (:,1:4);

% Do not include T-bills as an asset
number_of_assets = size(returns, 2);
risky_expected_returns = expReturns(1:4);
covMatrix_risky = cov(returns);

number_of_portfolios = 5000000;
results = zeros(3, number_of_portfolios);
weights_matrix = zeros(number_of_portfolios, number_of_assets);

for i = 1:number_of_portfolios
    weights = rand(number_of_assets, 1);
    weights = weights/sum(weights);
    portfolio_expected_return = weights' * risky_expected_returns';
    portfolio_risk = sqrt(weights' * covMatrix_risky * weights);

    results(1, i) = portfolio_expected_return;
    results(2, i) = portfolio_risk;

    % Sharpe ratio
    results(3, i) = portfolio_expected_return/portfolio_risk;

    weights_matrix(i, :) = weights';
end

[~, highest_index] = max(results(3, :));
optimal_weights = weights_matrix(highest_index, :);

disp('Optimal Weights for the Portfolio with the Highest Sharpe Ratio:'); 
for i = 1:number_of_assets
    fprintf('%s %.2f%%\n', portfoliocomponents(i,:), optimal_weights(i) * 100);
end

unique_returns = unique(round(results(1, :), 4));

efficient_frontier_volatility = zeros(size(unique_returns));
efficient_frontier_return = zeros(size(unique_returns));

for i = 1:length(unique_returns)
    % Index of portfolios with i return level
    index = find(round(results(1, :), 4) == unique_returns(i));

    % Find portfolio with lowest volatility
    [~, lowest_index] = min(results(2, index));
    efficient_frontier_volatility(i) = results(2, index(lowest_index));
    efficient_frontier_return(i) = unique_returns(i);
end

% Prevent negative returns
efficient_frontier_volatility(efficient_frontier_return < 0) = [];
efficient_frontier_return(efficient_frontier_return < 0) = [];

% Visualisation
figure4 = figure('Name','Markowitz Efficient Frontier');
scatter(efficient_frontier_volatility, efficient_frontier_return, 50, 'r', 'filled');
hold on;

xlabel('Volatility (Standard Deviation)');
ylabel('Return');
title('Markowitz Efficient Frontier');
legend('Efficient Frontier');
hold off;

% ---------- End of Markowitz Efficient Frontier ----------

%% ---------- Nelder-Mead method ----------
x0 = [0.2, 0.2, 0.2, 0.2, 0.2];

objective_function = @(weights) barrier_function(weights, sigma_p_max, ...
    proportion_cash_min, proportion_real_estate_min, ...
    proportion_equity_min, proportion_gold_min, proportion_tbill_min);

options_NM = optimset('Display', 'iter', 'FunValCheck', 'on', 'MaxIter', 100000, 'PlotFcns', @optimplotfval);
optimal_weights_NM = fminsearch(objective_function, x0, options_NM);
disp(sum(optimal_weights_NM))
disp('Optimal Weights from Nelder-Mead:');
disp(optimal_weights_NM);

% ---------- End of Nelder-Mead method ----------

%% ---------- SQP Method ----------
options_SQP = optimoptions('fmincon', 'SpecifyObjectiveGradient', true, 'Display', 'iter-detailed', 'Algorithm', 'sqp');

x0 = [0.15, 0.05, 0.5, 0.25, 0.05];
Aeq = [1, 1, 1, 1, 1];
beq = 1;
lb = [0.1, 0.1, 0.1, 0.1, 0.1];
ub = [0.6, 0.6, 0.6, 0.6, 0.6];
optimal_weights_SQP = fmincon(@obj_and_gradient, x0, [], [], Aeq, beq, lb, ub, @nonlinear_constraints, options_SQP);

% check if gradient calculation is correct
valid = checkGradients(@obj_and_gradient, x0, Display="on");

disp('Optimal Weights from SQP:');
disp(optimal_weights_SQP);

% ---------- end of SQP method ----------

%% ---------- Functions ----------

% ---------- Function for calculating return and volatility ----------

function [E_p,sigma_p] = calc(weightmatrix, expReturns, covMatrix)
    weightmatrix = weightmatrix(:);
    E_p = weightmatrix' * expReturns';
    sigma_p = sqrt(weightmatrix' * covMatrix * weightmatrix);
end

% ---------- Objective function ----------

function [S_p,U,weighted_U] = obj(E_p,sigma_p,E_risk_free,alpha)
    S_p = (E_p - E_risk_free)/sigma_p;
    U = E_p - alpha*(sigma_p^2);
    weighted_U = -(0.5 * S_p + 0.5 * U);
end

% ---------- Barrier function ----------

function F = barrier_function(weights, sigma_p_max, proportion_cash_min, proportion_real_estate_min, ...
    proportion_equity_min, proportion_gold_min, proportion_tbill_min)
    global expReturns E_risk_free covMatrix alpha

    weights = weights(:);


    [E_p, sigma_p] = calc(weights, expReturns, covMatrix);
    [~, ~, weighted_U] = obj(E_p, sigma_p, E_risk_free, alpha);

    g = [sigma_p / sigma_p_max - 1;
         proportion_cash_min - weights(1);
         proportion_real_estate_min - weights(2);
         proportion_equity_min - weights(3);
         proportion_gold_min - weights(4);
         proportion_tbill_min - weights(5)];

    % Weights constraint
    sum_weights_constraint = sum(weights) - 1;

    % Print intermediate values for debugging
    disp('Current Weights:');
    disp(weights');
    disp('Objective Value:');
    disp(weighted_U);

    % Prevent infeasible solution
    if any(g >= 0)
        % Penalty
        F = inf;
    else
        r = 100;
        F = weighted_U - (1 / r) * sum(log(-g)) + 1e4 * abs(sum_weights_constraint);
    end
end

% ---------- Creating objective and gradient ----------
function [f,g] = obj_and_gradient(x)
global expReturns E_risk_free covMatrix alpha

syms x1 x2 x3 x4 x5
assume(x1, "real")
assume(x2, "real")
assume(x3, "real")
assume(x4, "real")
assume(x5, "real")

weights = [x1; x2; x3; x4; x5];
[E_p,sigma_p] = calc(weights,expReturns,covMatrix);
[~,~,weighted_U_sym] = obj(E_p,sigma_p,E_risk_free,alpha);

% Substitute numeric values into the objective function
weighted_U_Numeric = subs(weighted_U_sym, ...
    [x1; x2; x3; x4; x5], ...
     [x(1); x(2); x(3); x(4); x(5)]) ;

f = double(weighted_U_Numeric);

Df_1 = diff(weighted_U_sym, x1);
Df_2 = diff(weighted_U_sym, x2);
Df_3 = diff(weighted_U_sym, x3);
Df_4 = diff(weighted_U_sym, x4);
Df_5 = diff(weighted_U_sym, x5);

% Substitute numeric values into the derivatives
Df_1Numeric = subs(Df_1, ...
    [x1; x2; x3; x4; x5], ...
     [x(1); x(2); x(3); x(4); x(5)]) ;
Df_1Value = double(Df_1Numeric);

Df_2Numeric = subs(Df_2, ...
    [x1; x2; x3; x4; x5], ...
     [x(1); x(2); x(3); x(4); x(5)]) ;
Df_2Value = double(Df_2Numeric);

Df_3Numeric = subs(Df_3, ...
    [x1; x2; x3; x4; x5], ...
     [x(1); x(2); x(3); x(4); x(5)]) ;
Df_3Value = double(Df_3Numeric);

Df_4Numeric = subs(Df_4, ...
    [x1; x2; x3; x4; x5], ...
     [x(1); x(2); x(3); x(4); x(5)]) ;
Df_4Value = double(Df_4Numeric);

Df_5Numeric = subs(Df_5, ...
    [x1; x2; x3; x4; x5], ...
     [x(1); x(2); x(3); x(4); x(5)]) ;
Df_5Value = double(Df_5Numeric);

g = [Df_1Value; Df_2Value; Df_3Value; Df_4Value; Df_5Value];

end

% ---------- Nonlinear constraints ----------
function [c,ceq] = nonlinear_constraints(x)
global expReturns E_risk_free covMatrix alpha sigma_p_max

syms x1 x2 x3 x4 x5
assume(x1, "real")
assume(x2, "real")
assume(x3, "real")
assume(x4, "real")
assume(x5, "real")

weights = [x1; x2; x3; x4; x5];
[E_p,sigma_p] = calc(weights,expReturns,covMatrix);

% Substitute numeric values into volatility function
sigma_p_Numeric = subs(sigma_p, ...
    [x1; x2; x3; x4; x5], ...
     [x(1); x(2); x(3); x(4); x(5)]) ;

c = (double(sigma_p_Numeric)/sigma_p_max) - 1;
ceq = [];
end

% ---------- End of Functions ----------
