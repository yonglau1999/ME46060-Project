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


% ---------- end of constraints ----------
%% ---------- Monotonicity ----------
syms y1 y3 y4 y5
assume(y1, "real")
assume(y3, "real")
assume(y4, "real")
assume(y5, "real")

y2 = 1 - y1 - y3 - y4 - y5;
weights = [y1; y2; y3; y4; y5];
[E_p,sigma_p] = calc(weights,expReturns,covMatrix);
[S_p_sym,U_sym,weighted_U_sym] = obj(E_p,sigma_p,E_risk_free,alpha);

Df_1 = vpa(diff(weighted_U_sym, y1));
Df_3 = vpa(diff(weighted_U_sym, y3));
Df_4 = vpa(diff(weighted_U_sym, y4));
Df_5 = vpa(diff(weighted_U_sym, y5));

y_values = linspace(0.1, 0.6, 500);

% Substitute numeric values into the Hessian matrix
Df_1Numeric = vpa(subs(Df_1, ...
    [y3; y4; y5], ...
    [0.1; 0.1; 0.1])) ;

Df_1_func = matlabFunction(Df_1Numeric);
Df_1_numbers = Df_1_func(y_values);

% Substitute numeric values into the Hessian matrix
Df_3Numeric = vpa(subs(Df_3, ...
    [y1; y4; y5], ...
    [0.1; 0.1; 0.1])) ;

Df_3_func = matlabFunction(Df_3Numeric);
Df_3_numbers = Df_3_func(y_values);

% Substitute numeric values into the Hessian matrix
Df_4Numeric = vpa(subs(Df_4, ...
    [y1; y3; y5], ...
    [0.1; 0.1; 0.1])) ;

Df_4_func = matlabFunction(Df_4Numeric);
Df_4_numbers = Df_4_func(y_values);

% Substitute numeric values into the Hessian matrix
Df_5Numeric = vpa(subs(Df_5, ...
    [y1; y3; y4], ...
    [0.1; 0.1; 0.1])) ;

Df_5_func = matlabFunction(Df_5Numeric);
Df_5_numbers = Df_5_func(y_values);

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
%% Define the symbolic variables
syms x1 x2 x3 x4 x5

weights = [x1; x2; x3; x4; x5];

expReturnsSym = sym('expReturns', [1, 5]);

covMatrixSym = sym('covMatrix', [5, 5]);

[preturn_sym,pvolatility_sym] = calc(weights,expReturnsSym,covMatrixSym);

E_risk_free_sym = sym('E_risk_free');

alpha_sym = sym('alpha');

[S_p_sym,U_sym,weighted_U_sym] = obj(preturn_sym,pvolatility_sym,E_risk_free_sym,alpha_sym);

HessianMatrix = hessian(weighted_U_sym, weights);

weight = [0.2;0.2;0.2;0.2;0.2]; % Change this weightage to check for a non-convex solution

% Substitute numeric values into the Hessian matrix
HessianMatrixNumeric = vpa(subs(HessianMatrix, ...
    [expReturnsSym(:); covMatrixSym(:); E_risk_free_sym; alpha_sym ;weights(:)], ...
    [expReturns(:); covMatrix(:); E_risk_free; alpha ; weight(:)])) ;


eigenValues = eig(HessianMatrixNumeric);

% If all eigenvalues are non-negative, the function is convex
isConvex = all(eigenValues >= 0);

if isConvex
    disp('The objective function is convex.');
else
    disp('The objective function is not convex.');
end


% %% ---------- Own Optimiser ----------
% numAssets = size(returns, 2);  % Number of assets
% expReturns = mean(returns);    % Expected returns of each asset
% covMatrix = cov(returns);      % Covariance matrix of the returns
% 
% 
% numPortfolios = 1000000; % Define the number of portfolios to simulate
% results = zeros(3, numPortfolios);  % Initialize results matrix
% weightsMatrix = zeros(numPortfolios, numAssets);  % To store weights of each portfolio
% 
% for i = 1:numPortfolios
%     weights = rand(numAssets, 1);  % Randomly generate weights
%     weights = weights / sum(weights);  % Normalize weights to sum to 1
%     portfolioReturn = weights' * expReturns';  % Expected return of the portfolio
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
% 
% uniqueReturns = unique(round(results(1, :), 4));% Find unique levels of return and round to 4 decimal places
% 
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
% % Fit a quadratic curve to the efficient frontier data
% p = polyfit(efficientVolatility, efficientReturn, 3); % Use 'polyfit' function with degree 3 
% 
% % Generate a range of volatility values for plotting the fitted curve
% volatilityRange = linspace(min(efficientVolatility), max(efficientVolatility), 100);
% 
% % Calculate the corresponding return values using the fitted equation
% fittedReturn = polyval(p, volatilityRange); % Use 'polyval' function to evaluate the fitted polynomial
% 
% % Plot the efficient frontier and the fitted curve
% figure(1);
% scatter(efficientVolatility, efficientReturn, 50, 'r', 'filled'); % Plot the efficient frontier
% hold on;
% plot(volatilityRange, fittedReturn, 'b-', 'LineWidth', 2); % Plot the fitted curve
% xlabel('Volatility (Standard Deviation)');
% ylabel('Return');
% title('Efficient Frontier with Fitted Curve');
% legend('Efficient Frontier', 'Fitted Curve');
% hold off;
% 
% disp('Equation of the fitted curve (efficient frontier):');
% disp(['f(volatility) = ', num2str(p(1)), ' * volatility^3 + ', num2str(p(2)), ' * volatility^2 + '...
%     , num2str(p(3)),'* volatility + ',num2str(p(4))]);
%% ---------- Function for calculating return and volatility ----------
function [E_p,sigma_p] = calc(weightmatrix,expReturns,covMatrix)
    E_p = weightmatrix' * expReturns';
    sigma_p = sqrt(weightmatrix' * covMatrix * weightmatrix);
end

%% ---------- Objective functions ----------

%Sharpe Ratio and Utility
function [S_p,U,weighted_U] = obj(E_p,sigma_p,E_risk_free,alpha)
    S_p = (E_p - E_risk_free)/sigma_p;
    U = E_p - alpha*(sigma_p^2);
    weighted_U = -(0.5 * S_p + 0.5 * U);
end
% ---------- end of objective functions ----------

