%% only change the variables in the design variables and constraints section

%% ---------- Load Returns data ----------
portfoliocomponents=char({'Cash','RealEstate','Equity','Gold','Tbill'});
returnsTable=readtable('Compiled_Returns_Data.csv');
returns = table2array(returnsTable(:,1:5));

expReturns = mean(returns);
E_risk_free=expReturns(1,5);
covMatrix = cov(returns);

%% ---------- design variables ----------

proportion_cash = 0.2;
proportion_equity = 0.2;
proportion_gold =0.2;
proportion_tbill = 0.2;

alpha=0.5;

% ---------- end of design variables ----------
%% ---------- constraints ----------

% Constraint 1: Sum of proportions = 1
proportion_real_estate = 1 - proportion_equity - proportion_tbill - proportion_gold - proportion_cash;




% Constraint 2: Maximum volatility
sigma_p_max = 0;


% Constraint 3 : Diversification
proportion_cash_min = 0.1;
proportion_real_estate_min = 0.1;
proportion_equity_min = 0.1;
proportion_gold_min = 0.1;
proportion_tbill_min = 0.1;

% Weights

weights = [proportion_cash;proportion_real_estate;proportion_equity;proportion_gold;proportion_tbill]

% ---------- end of constraints ----------
%% ---------- Monotonicity ----------
% weighted_U = -(0.5 * ([x1; x2; x3; x4; x5]' * (expReturns' - E_risk_free) / ...
%     sqrt([x1; x2; x3; x4; x5]' * covMatrix * [x1; x2; x3; x4; x5])) + ...
%     0.5 * ([x1; x2; x3; x4; x5]' * expReturns' - alpha * ([x1; x2; x3; x4; x5]' * covMatrix * [x1; x2; x3; x4; x5])));
syms x1 x2 x3 x4;
objective_function_Df_x1 = diff(objective_function(x1, x2, x3, x4));

%% ---------- Get return, volatility ----------
[E_p,sigma_p] = calc([proportion_equity;proportion_tbill;proportion_gold;proportion_cash;proportion_real_estate],expReturns,...
    covMatrix);

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
%     [preturn,pvolatility] = calc(weights,expReturns,covMatrix);
%     [S_p,U,weighted_U] = obj(preturn,pvolatility,E_risk_free,alpha);
% 
% 
%     results(1, i) = preturn;  % Store portfolio return
%     results(2, i) = pvolatility;    % Store portfolio risk
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


%% ---------- Own Optimiser ----------
returns = returns (:,1:4);
numAssets = size(returns, 2);  % Number of assets, removing T-bills (Non risky)
expReturns = mean(returns);    % Expected returns of each asset
covMatrix = cov(returns);      % Covariance matrix of the returns


numPortfolios = 5000000; % Define the number of portfolios to simulate
results = zeros(3, numPortfolios);  % Initialize results matrix
weightsMatrix = zeros(numPortfolios, numAssets);  % To store weights of each portfolio

for i = 1:numPortfolios
    weights = rand(numAssets, 1);  % Randomly generate weights
    weights = weights / sum(weights);  % Normalize weights to sum to 1
    portfolioReturn = weights' * expReturns';  % Expected return of the portfolio
    portfolioRisk = sqrt(weights' * covMatrix * weights);  % Risk (standard deviation) of the portfolio

    results(1, i) = portfolioReturn;  % Store portfolio return
    results(2, i) = portfolioRisk;    % Store portfolio risk
    results(3, i) = portfolioReturn / portfolioRisk;  % Store Sharpe ratio (return/risk)

    weightsMatrix(i, :) = weights';  % Store the weights
end

[~, maxIndex] = max(results(3, :)) % Identify the portfolio with the highest Sharpe ratio
optimalWeights = weightsMatrix(maxIndex, :);

disp('Optimal Weights for the Portfolio with the Highest Sharpe Ratio:'); 
for i = 1:numAssets
    fprintf('%s %.2f%%\n', portfoliocomponents(i,:), optimalWeights(i) * 100);
end

uniqueReturns = unique(round(results(1, :), 4));% Find unique levels of return and round to 4 decimal places

efficientVolatility = zeros(size(uniqueReturns));% Initialize arrays to store volatility and return for the efficient frontier
efficientReturn = zeros(size(uniqueReturns));

for i = 1:length(uniqueReturns)
    % Find indices of portfolios with the current return level
    indices = find(round(results(1, :), 4) == uniqueReturns(i));

    % Determine the portfolio with the lowest volatility among these
    [~, minIndex] = min(results(2, indices));
    efficientVolatility(i) = results(2, indices(minIndex));
    efficientReturn(i) = uniqueReturns(i);
end

% Exclude return levels below 0
efficientVolatility(efficientReturn < 0) = [];
efficientReturn(efficientReturn < 0) = [];

% Plot the efficient frontier and the fitted curve
figure(1);
scatter(efficientVolatility, efficientReturn, 50, 'r', 'filled'); % Plot the efficient frontier
hold on;

xlabel('Volatility (Standard Deviation)');
ylabel('Return');
title('Efficient Frontier');
legend('Efficient Frontier');
hold off;
%% ---------- Function for calculating return and volatility ----------
<<<<<<< HEAD
function [E_p,sigma_p] = calc(weightmatrix,expReturns,covMatrix)
    E_p = weightmatrix' * expReturns';
    sigma_p = (weightmatrix' * covMatrix * weightmatrix);
=======
function [preturn,pvolatility] = calc(weightmatrix,expReturns,covMatrix)
    preturn = weightmatrix' * expReturns';
    pvolatility = sqrt(weightmatrix' * covMatrix * weightmatrix);
>>>>>>> 78388c99a96e347392fb1872ae1d9de8dcd9075a
end

%% ---------- Objective functions ----------

%Sharpe Ratio and Utility
function [S_p,U,weighted_U] = obj(E_p,sigma_p,E_risk_free,alpha)
    S_p = (E_p - E_risk_free)/sigma_p;
    U = E_p - alpha*(sigma_p^2);
    weighted_U = -(0.5 * S_p + 0.5 * U);
end

function obj = objective_function(x1, x2, x3, x4)
    obj = -(0.5 * ([x1; x2; x3; x4; 1 - x1 - x2- x3 - x4]' * (expReturns' - E_risk_free) / ...
    sqrt([x1; x2; x3; x4; 1 - x1 - x2- x3 - x4]' * covMatrix * [x1; x2; x3; x4; 1 - x1 - x2- x3 - x4])) + ...
    0.5 * ([x1; x2; x3; x4; 1 - x1 - x2- x3 - x4]' * expReturns' - alpha * ([x1; x2; x3; x4; 1 - x1 - x2- x3 - x4]' * covMatrix * [x1; x2; x3; x4; 1 - x1 - x2- x3 - x4])));
end
% ---------- end of objective functions ----------

