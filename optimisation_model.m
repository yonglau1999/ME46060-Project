%% only change the variables in the design variables and constraints section

%% ---------- Load Returns data ----------
portfoliocomponents=char({'Cash','RealEstate','Equity','Gold','Tbill'});
returnsTable=readtable('Compiled_Returns_Data.csv');
returns = table2array(returnsTable(:,1:5));

global expReturns E_risk_free covMatrix;
expReturns = [0.002, 0.096, 0.113, 0.106, 0.03];
E_risk_free = 0.03;
covMatrix = cov(returns);

% ---------- end of design variables ----------

%% ---------- Constraints ----------

% Constraint 1: Sum of proportions = 1
proportion_real_estate = 1 - proportion_equity - proportion_tbill - proportion_gold - proportion_cash;


% Constraint 2: Maximum volatility
sigma_p_max = 0.1575;


% Constraint 3 : Diversification
proportion_cash_min = 0.1;
proportion_real_estate_min = 0.1;
proportion_equity_min = 0.1;
proportion_gold_min = 0.1;
proportion_tbill_min = 0.1;

% ---------- end of constraints ----------

%% ---------- Constants ----------
global alpha
alpha = 0.5;
% ---------- end of constants ----------
%% ---------- Objective function against weightage ----------
% syms y1 y3 y4 y5
% assume(y1, "real")
% assume(y3, "real")
% assume(y4, "real")
% assume(y5, "real")
% 
% y2 = 1 - y1 - y3 - y4 - y5;
% weights = [y1; y2; y3; y4; y5];
% [E_p,sigma_p] = calc(weights,expReturns,covMatrix);
% [S_p_sym,U_sym,weighted_U_sym] = obj(E_p,sigma_p,E_risk_free,alpha);
% 
% y_values = linspace(0.1, 0.6, 500);
% 
% % Substitute numeric values into the Hessian matrix
% Df_1Numeric = vpa(subs(weighted_U_sym, ...
%     [y3; y4; y5], ...
%     [0.1; 0.1; 0.1])) ;
% 
% Df_1_func = matlabFunction(Df_1Numeric);
% Df_1_numbers = Df_1_func(y_values);
% 
% % Substitute numeric values into the Hessian matrix
% Df_3Numeric = vpa(subs(weighted_U_sym, ...
%     [y1; y4; y5], ...
%      [0.1; 0.1; 0.1])) ;
% 
% Df_3_func = matlabFunction(Df_3Numeric);
% Df_3_numbers = Df_3_func(y_values);
% 
% % Substitute numeric values into the Hessian matrix
% Df_4Numeric = vpa(subs(weighted_U_sym, ...
%     [y1; y3; y5], ...
%      [0.1; 0.1; 0.1])) ;
% 
% Df_4_func = matlabFunction(Df_4Numeric);
% Df_4_numbers = Df_4_func(y_values);
% 
% % Substitute numeric values into the Hessian matrix
% Df_5Numeric = vpa(subs(weighted_U_sym, ...
%     [y1; y3; y4], ...
%      [0.1; 0.1; 0.1])) ;
% 
% Df_5_func = matlabFunction(Df_5Numeric);
% Df_5_numbers = Df_5_func(y_values);
% 
% subplot(2, 2, 1)
% plot(y_values, Df_1_numbers)
% xlabel('Weightage') 
% ylabel('Objective Function') 
% title('Objective function value with respect to proportion of cash')
% subplot(2, 2, 2)
% plot(y_values, Df_3_numbers)
% xlabel('Weightage') 
% ylabel('Objective Function') 
% title('Objective function value with respect to proportion of equity')
% subplot(2, 2, 3)
% plot(y_values, Df_4_numbers)
% xlabel('Weightage') 
% ylabel('Objective Function')  
% title('Objective function value with respect to proportion of gold')
% subplot(2, 2, 4)
% plot(y_values, Df_5_numbers)
% xlabel('Weightage') 
% ylabel('Objective Function')   
% title('Objective function value with respect to proportion of Tbill')

% ---------- end of objective function against weightage ----------

%% ---------- Monotonicity ----------
% syms y1 y3 y4 y5
% assume(y1, "real")
% assume(y3, "real")
% assume(y4, "real")
% assume(y5, "real")
% 
% y2 = 1 - y1 - y3 - y4 - y5;
% weights = [y1; y2; y3; y4; y5];
% [E_p,sigma_p] = calc(weights,expReturns,covMatrix);
% [S_p_sym,U_sym,weighted_U_sym] = obj(E_p,sigma_p,E_risk_free,alpha);
% 
% Df_1 = vpa(diff(weighted_U_sym, y1));
% Df_3 = vpa(diff(weighted_U_sym, y3));
% Df_4 = vpa(diff(weighted_U_sym, y4));
% Df_5 = vpa(diff(weighted_U_sym, y5));
% 
% y_values = linspace(0.1, 0.2, 500);
% 
% % Substitute numeric values into the Hessian matrix
% Df_1Numeric = vpa(subs(Df_1, ...
%     [y3; y4; y5], ...
%     [0.1; 0.1; 0.1])) ;
% 
% Df_1_func = matlabFunction(Df_1Numeric);
% Df_1_numbers = Df_1_func(y_values);
% 
% % Substitute numeric values into the Hessian matrix
% Df_3Numeric = vpa(subs(Df_3, ...
%     [y1; y4; y5], ...
%     [0.2; 0.2; 0.2])) ;
% 
% Df_3_func = matlabFunction(Df_3Numeric);
% Df_3_numbers = Df_3_func(y_values);
% 
% % Substitute numeric values into the Hessian matrix
% Df_4Numeric = vpa(subs(Df_4, ...
%     [y1; y3; y5], ...
%     [0.2; 0.2; 0.2])) ;
% 
% Df_4_func = matlabFunction(Df_4Numeric);
% Df_4_numbers = Df_4_func(y_values);
% 
% % Substitute numeric values into the Hessian matrix
% Df_5Numeric = vpa(subs(Df_5, ...
%     [y1; y3; y4], ...
%     [0.2; 0.2; 0.2])) ;
% 
% Df_5_func = matlabFunction(Df_5Numeric);
% Df_5_numbers = Df_5_func(y_values);
% 
% subplot(2, 2, 1)
% plot(y_values, Df_1_numbers)
% xlabel('Proportion of Cash') 
% ylabel('Gradient') 
% title('Gradient with respect to proportion of cash')
% subplot(2, 2, 2)
% plot(y_values, Df_3_numbers)
% xlabel('Proportion of Equity') 
% ylabel('Gradient') 
% title('Gradient with respect to proportion of equity')
% subplot(2, 2, 3)
% plot(y_values, Df_4_numbers)
% xlabel('Proportion of Gold') 
% ylabel('Gradient') 
% title('Gradient with respect to proportion of gold')
% subplot(2, 2, 4)
% plot(y_values, Df_5_numbers)
% xlabel('Proportion of Tbill') 
% ylabel('Gradient') 
% title('Gradient with respect to proportion of Tbill')

% ---------- end of monotonicity ----------
%% ---------- Design sensitivities ----------

% syms z1 z3 z4 z5
% assume(z1, "real")
% assume(z3, "real")
% assume(z4, "real")
% assume(z5, "real")
% 
% z2 = 1 - z1 - z3 - z4 - z5;
% weights = [z1; z2; z3; z4; z5];
% [E_p,sigma_p] = calc(weights,expReturns,covMatrix);
% [S_p_sym,U_sym,weighted_U_sym] = obj(E_p,sigma_p,E_risk_free,alpha);
% 
% 
% 
% Df_1 = vpa((z1/weighted_U_sym)*diff(weighted_U_sym, z1));
% Df_3 = vpa((z3/weighted_U_sym)*diff(weighted_U_sym, z3));
% Df_4 = vpa((z4/weighted_U_sym)*diff(weighted_U_sym, z4));
% Df_5 = vpa((z5/weighted_U_sym)*diff(weighted_U_sym, z5));
% 
% z_values = linspace(0.1, 0.2, 500);
% 
% % Substitute numeric values into the Hessian matrix
% Df_1Numeric = vpa(subs(Df_1, ...
%     [z3; z4; z5], ...
%     [0.2; 0.2; 0.2])) ;
% 
% Df_1_func = matlabFunction(Df_1Numeric);
% Df_1_numbers = Df_1_func(z_values);
% 
% % Substitute numeric values into the Hessian matrix
% Df_3Numeric = vpa(subs(Df_3, ...
%     [z1; z4; z5], ...
%     [0.2 ;0.2; 0.2])) ;
% 
% Df_3_func = matlabFunction(Df_3Numeric);
% Df_3_numbers = Df_3_func(z_values);
% 
% % Substitute numeric values into the Hessian matrix
% Df_4Numeric = vpa(subs(Df_4, ...
%     [z1; z3; z5], ...
%     [0.2; 0.2; 0.2])) ;
% 
% Df_4_func = matlabFunction(Df_4Numeric);
% Df_4_numbers = Df_4_func(z_values);
% 
% % Substitute numeric values into the Hessian matrix
% Df_5Numeric = vpa(subs(Df_5, ...
%     [z1; z3; z4], ...
%     [0.2; 0.2; 0.2])) ;
% 
% Df_5_func = matlabFunction(Df_5Numeric);
% Df_5_numbers = Df_5_func(z_values);
% 
% subplot(2, 2, 1)
% plot(z_values, Df_1_numbers)
% xlabel('Proportion of Cash') 
% ylabel('Log Gradient') 
% title('Log Gradient with respect to proportion of cash')
% subplot(2, 2, 2)
% plot(z_values, Df_3_numbers)
% xlabel('Proportion of Equity') 
% ylabel('Log Gradient') 
% title('Log Gradient with respect to proportion of equity')
% subplot(2, 2, 3)
% plot(z_values, Df_4_numbers)
% xlabel('Proportion of Gold') 
% ylabel('Log Gradient') 
% title('Log Gradient with respect to proportion of gold')
% subplot(2, 2, 4)
% plot(z_values, Df_5_numbers)
% xlabel('Proportion of Tbill') 
% ylabel('Log Gradient') 
% title('Log Gradient with respect to proportion of Tbill')

% ---------- end of design sensitivities ----------

%% ---------- Boundedness Check ----------
% 
% numAssets = size(returns, 2);
% numPortfolios = 1000000; % Define the number of portfolios to simulate
% results = zeros(3, numPortfolios);  % Initialize results matrix
% weightsMatrix = zeros(numPortfolios, numAssets);  % To store weights of each portfolio
% 
% for i = 1:numPortfolios
%     weights = rand(numAssets, 1);  % Randomly generate weights
%     weights = weights / sum(weights);  % Normalize weights to sum to 1
%     [E_P,sigma_p] = calc(weights,expReturns,covMatrix);
%     [S_p,U,weighted_U] = obj(E_P,sigma_p,E_risk_free,alpha);
% 
% 
%     results(1, i) = E_P;  % Store portfolio return
%     results(2, i) = sigma_p;    % Store portfolio risk
%     results(3, i) = weighted_U;  % Store weighted objective
% 
%     weightsMatrix(i, :) = weights';  % Store the weights
% end
% 
% figure(1)
% scatter(results(2,:),results(3,:),50,"red")
% xlabel('Volatility')
% ylabel('Objective function')
% 
% 
% 
% [minU,index]=min(results(3,:));
% corrweights=weightsMatrix(index,:);

% ---------- end of boundedness check ----------
%% ---------- Convexity ----------
% syms x1 x2 x3 x4 x5
% 
% weights = [x1; x2; x3; x4; x5];
% 
% [E_p_sym,sigma_p_sym] = calc(weights,expReturns,covMatrix);
% 
% [S_p_sym,U_sym,weighted_U_sym] = obj(E_p_sym,sigma_p_sym,E_risk_free,alpha);
% 
% HessianMatrix = hessian(weighted_U_sym, weights)
% 
% weight = [0.2;0.2;0.2;0.2;0.2]; % Change this weightage to check for a non-convex solution
% 
% % Substitute numeric values into the Hessian matrix
% HessianMatrixNumeric = vpa(subs(HessianMatrix, ...
%     [weights(:)], ...
%     [weight(:)])) ;
% 
% 
% eigenValues = eig(HessianMatrixNumeric);
% 
% % If all eigenvalues are non-negative, the function is convex
% isConvex = all(eigenValues >= 0);
% 
% if isConvex
%     disp('The objective function is convex.');
% else
%     disp('The objective function is not convex.');
% end

% ---------- end of convexity ----------
%% ---------- Markowitz Efficient Frontier ----------
% returns = returns (:,1:4);
% numAssets = size(returns, 2);  % Number of assets, removing T-bills (Non risky)
% expReturns1 = expReturns(1:4);    % Expected returns of each asset
% covMatrix = cov(returns);      % Covariance matrix of the returns
% 
% 
% numPortfolios = 5000000; % Define the number of portfolios to simulate
% results = zeros(3, numPortfolios);  % Initialize results matrix
% weightsMatrix = zeros(numPortfolios, numAssets);  % To store weights of each portfolio
% 
% for i = 1:numPortfolios
%     weights = rand(numAssets, 1);  % Randomly generate weights
%     weights = weights / sum(weights);  % Normalize weights to sum to 1
%     portfolioReturn = weights' * expReturns1';  % Expected return of the portfolio
%     portfolioRisk = sqrt(weights' * covMatrix * weights);  % Risk (standard deviation) of the portfolio
% 
%     results(1, i) = portfolioReturn;  % Store portfolio return
%     results(2, i) = portfolioRisk;    % Store portfolio risk
%     results(3, i) = portfolioReturn / portfolioRisk;  % Store Sharpe ratio (return/risk)
% 
%     weightsMatrix(i, :) = weights';  % Store the weights
% end
% 
% [~, maxIndex] = max(results(3, :)) % Identify the portfolio with the highest Sharpe ratio
% optimalWeights = weightsMatrix(maxIndex, :);
% 
% disp('Optimal Weights for the Portfolio with the Highest Sharpe Ratio:'); 
% for i = 1:numAssets
%     fprintf('%s %.2f%%\n', portfoliocomponents(i,:), optimalWeights(i) * 100);
% end
% 
% uniqueReturns = unique(round(results(1, :), 4));% Find unique levels of return and round to 4 decimal places
% 
% efficientVolatility = zeros(size(uniqueReturns));% Initialize arrays to store volatility and return for the efficient frontier
% efficientReturn = zeros(size(uniqueReturns));
% 
% for i = 1:length(uniqueReturns)
%     % Find indices of portfolios with the current return level
%     indices = find(round(results(1, :), 4) == uniqueReturns(i));
% 
%     % Determine the portfolio with the lowest volatility among these
%     [~, minIndex] = min(results(2, indices));
%     efficientVolatility(i) = results(2, indices(minIndex));
%     efficientReturn(i) = uniqueReturns(i);
% end
% 
% % Exclude return levels below 0
% efficientVolatility(efficientReturn < 0) = [];
% efficientReturn(efficientReturn < 0) = [];
% 
% % Plot the efficient frontier and the fitted curve
% figure(1);
% scatter(efficientVolatility, efficientReturn, 50, 'r', 'filled'); % Plot the efficient frontier
% hold on;
% 
% xlabel('Volatility (Standard Deviation)');
% ylabel('Return');
% title('Efficient Frontier');
% legend('Efficient Frontier');
% hold off;

% ---------- end of markowitz efficient frontier ----------
%% ---------- Nelder-Mead method ----------
% x0 = [0.2, 0.2, 0.2, 0.2,0.2]; % Starting point
% 
% % Define the objective function handle
% objective_function = @(weights) barrier_function(weights, sigma_p_max, ...
%     proportion_cash_min, proportion_real_estate_min, ...
%     proportion_equity_min, proportion_gold_min, proportion_tbill_min);
% 
% % Call fminsearch
% options = optimset('Display', 'iter', 'FunValCheck', 'on', 'MaxIter', 100000, 'PlotFcns', @optimplotfval,'TolX',1e-20,'TolFun',1e-20);
% optimal_weights = fminsearch(objective_function, x0, options);
% disp(sum(optimal_weights))
% disp('Optimal Weights:');
% disp(optimal_weights);

% ---------- end of nelder-mead method ----------
%% ---------- SQP Method ----------
options = optimoptions('fmincon', 'SpecifyObjectiveGradient', true, 'Display', 'iter-detailed');

x0 = [0.15, 0.05, 0.5, 0.25, 0.05];
Aeq = [1, 1, 1, 1, 1];
beq = 1;
lb = [0.1, 0.1, 0.1, 0.1, 0.1];
ub = [0.6, 0.6, 0.6, 0.6, 0.6];
x = fmincon(@rosentwo, x0, [], [], Aeq, beq, lb, ub, @nonlinearConstraints, options);

valid = checkGradients(@rosentwo,x0,Display="on");

disp('Optimal Weights:');
disp(x);

% ---------- end of SQP method ----------
%% ---------- Function for calculating return and volatility ----------

function [E_p,sigma_p] = calc(weightmatrix,expReturns,covMatrix)
    weightmatrix = weightmatrix(:);
    E_p = weightmatrix' * expReturns';
    sigma_p = sqrt(weightmatrix' * covMatrix * weightmatrix);

end

%% ---------- Objective function ----------

%Sharpe Ratio and Utility
function [S_p,U,weighted_U] = obj(E_p,sigma_p,E_risk_free,alpha)
    S_p = (E_p - E_risk_free)/sigma_p;
    U = E_p - alpha*(sigma_p^2);
    weighted_U = -(0.5 * S_p + 0.5 * U);
end

% ---------- end of objective function----------
%% ---------- Creating barrier function ----------

function F = barrier_function(weights, sigma_p_max, proportion_cash_min, proportion_real_estate_min, ...
    proportion_equity_min, proportion_gold_min, proportion_tbill_min)
    global expReturns E_risk_free covMatrix alpha

    % Ensure weights are treated as a column vector
    weights = weights(:);

    % Calculate expected portfolio return and volatility
    [E_p, sigma_p] = calc(weights, expReturns, covMatrix);

    % Calculate Sharpe ratio and utility
    [~, ~, weighted_U] = obj(E_p, sigma_p, E_risk_free, alpha);

    % Define inequality constraints
    g = [sigma_p / sigma_p_max - 1;
         proportion_cash_min - weights(1);
         proportion_real_estate_min - weights(2);
         proportion_equity_min - weights(3);
         proportion_gold_min - weights(4);
         proportion_tbill_min - weights(5)];

    % Sum of weights constraint
    sum_weights_constraint = sum(weights) - 1;

    % Print intermediate values for debugging
    disp('Current Weights:');
    disp(weights');
    disp('Objective Value:');
    disp(weighted_U);

    % Check if any constraints are violated
    if any(g >= 0)
        F = inf; % Return a large penalty if any constraint is violated
    else
        % Define the barrier function
        r = 100; % Barrier constant for inequality constraint
        F = weighted_U - (1 / r) * sum(log(-g)) + 1e4 * abs(sum_weights_constraint);
    end
end

% ---------- end of barrier function against weightage ----------
%% ---------- Creating objective and gradient ----------
function [f,g] = rosentwo(x)
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

weighted_U_Numeric = subs(weighted_U_sym, ...
    [x1; x2; x3; x4; x5], ...
     [x(1); x(2); x(3); x(4); x(5)]) ;

f = double(weighted_U_Numeric);

Df_1 = diff(weighted_U_sym, x1);
Df_2 = diff(weighted_U_sym, x2);
Df_3 = diff(weighted_U_sym, x3);
Df_4 = diff(weighted_U_sym, x4);
Df_5 = diff(weighted_U_sym, x5);

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

% ---------- end of objective function against weightage ----------
%% ---------- Nonlinear constraints ----------
function [c, ceq] = nonlinearConstraints(x)
    global expReturns covMatrix sigma_p_max
    
    weightMatrix = [x(1); x(2); x(3); x(4); x(5)];
    [~, sigma_p] = calc(weightMatrix, expReturns, covMatrix);
    c = sigma_p - sigma_p_max;
    ceq = [];
end

% ---------- end of nonlinear constraints ----------