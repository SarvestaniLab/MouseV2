function [fitresult, gof] = EMcheck(Ecccell, Altcell, SFcell)
% Historical censored-plane fit used for submitted Figure 3F/G.
% Restored from Git commit eb4d306 without changing its equations.

x = Ecccell;
y = Altcell;
z = SFcell;
censored_below = z <= 0.01;
censored_above = z >= 3.99;
a = 0; b = 0; c = 0; sigma = 1;
max_iters = 100; tol = 1e-6;
n = length(z);

for iter = 1:max_iters
    mu = a * x + b * y + c;
    z_imputed = z;
    phi_below = normpdf(0, mu(censored_below), sigma);
    Phi_below = normcdf(0, mu(censored_below), sigma);
    z_imputed(censored_below) = mu(censored_below) - ...
        sigma .* (phi_below ./ (1 - Phi_below));
    phi_above = normpdf(4, mu(censored_above), sigma);
    Phi_above = normcdf(4, mu(censored_above), sigma);
    z_imputed(censored_above) = mu(censored_above) + ...
        sigma .* (phi_above ./ Phi_above);
    A = [x', y', ones(n,1)];
    coeffs = A \ z_imputed';
    a = coeffs(1); b = coeffs(2); c = coeffs(3);
    residuals = z_imputed - (a * x + b * y + c);
    sigma_new = sqrt(mean(residuals.^2));
    if abs(sigma - sigma_new) < tol
        break;
    end
    sigma = sigma_new;
end

fitresult = @(X,Y) a * X + b * Y + c;
gof.sigma = sigma;
gof.res = residuals;
