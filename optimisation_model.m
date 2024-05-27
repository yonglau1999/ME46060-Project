%% only change the variables in the design variables and constraints section

%% ---------- design variables ----------
proportion_equity = 0;
proportion_bonds = 0;
proportion_crypto = 0;
proportion_cash = 0;

% ---------- end of design variables ----------

%% ---------- constraints ----------

% proportion = 1
proportion_real_estate = 1 - proportion_equity - proportion_bonds - proportion_crypto - proportion_cash;

% maximum volatility
sigma_p_max = 0;

% minimum return
E_p_min = 0;

% efficient frontier of risky investments


% diversification
proportion_equity_min = 0.1;
proportion_bonds_min = 0.1;
proportion_crypto_min = 0.1;
proportion_cash_min = 0.1;
proportion_real_estate_min = 0.1;

% ---------- end of constraints ----------

%% ---------- objective functions ----------

% sharpe ratio
S_p = (E_p - E_risk_free)/sigma_p;

% utility
U = E_p - alpha*(sigma_p^2);

% ---------- end of objective functions ----------

